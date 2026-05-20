import argparse
import json
import os
import random
from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.EI import load_train_records
from cs336_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
)
from cs336_alignment.sft_helper import (
    get_response_log_probs,
    tokenize_prompt_and_output,
)
from cs336_alignment.sft_train import (
    GSM8K_TEST,
    MODEL_PATH,
    PROMPT_TEMPLATE_PATH,
    init_vllm,
    load_policy_into_vllm_instance,
    make_eval_split,
    run_eval,
)


def grpo_rollout(
    llm: LLM,
    prompts: list[str],
    group_size: int,
    temperature: float,
    min_tokens: int,
    max_tokens: int,
) -> list[list[str]]:
    sampling = SamplingParams(
        n=group_size,
        temperature=temperature,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling)
    return [[c.text for c in o.outputs] for o in outputs]


def plot_eval_rewards(eval_log_path: str, plot_path: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping plot at {plot_path}")
        return

    steps, ans, fmt = [], [], []
    with open(eval_log_path) as f:
        for line in f:
            r = json.loads(line)
            steps.append(r["grpo_step"])
            ans.append(r["avg_answer_reward"])
            fmt.append(r["avg_format_reward"])

    if not steps:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, ans, marker="o", label="avg_answer_reward")
    ax.plot(steps, fmt, marker="s", label="avg_format_reward", alpha=0.6)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("validation reward")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def grpo_step(
    *,
    policy,
    tokenizer,
    llm: LLM,
    train_records: list[dict],
    template: str,
    rng: random.Random,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rollout_batch_size: int,
    group_size: int,
    sampling_temperature: float,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    epochs_per_rollout_batch: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    use_std_normalization: bool,
    advantage_eps: float,
    cliprange: float,
    max_seq_len: int,
    max_grad_norm: float,
):
    assert rollout_batch_size % group_size == 0
    assert train_batch_size % gradient_accumulation_steps == 0
    assert train_batch_size == rollout_batch_size, (
        "On-policy: train_batch_size must equal rollout_batch_size"
    )
    micro_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts = rollout_batch_size // group_size

    # 1) Sample prompts and ground truths.
    sampled = rng.sample(train_records, n_prompts)
    questions = [r["question"] for r in sampled]
    golds = [r["gold"] for r in sampled]
    prompts = [template.format(question=q) for q in questions]

    # 2) Generate rollouts via vLLM.
    policy.eval()
    load_policy_into_vllm_instance(policy, llm)
    rollouts = grpo_rollout(
        llm,
        prompts,
        group_size=group_size,
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
    )
    policy.train()

    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    flat_gts: list[str] = []
    for p, gold, rs in zip(prompts, golds, rollouts, strict=True):
        for r in rs:
            flat_prompts.append(p)
            flat_responses.append(r)
            flat_gts.append(gold)
    assert len(flat_responses) == rollout_batch_size

    # 3) Group-normalized advantages.
    advantages, raw_rewards, reward_meta = compute_group_normalized_reward(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=flat_responses,
        repeated_ground_truths=flat_gts,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=use_std_normalization,
    )
    advantages = advantages.to(device)
    raw_rewards_dev = raw_rewards.to(device)

    # 4) Tokenize all (prompt, response) pairs once.
    toks = tokenize_prompt_and_output(flat_prompts, flat_responses, tokenizer)
    input_ids_all = toks["input_ids"][:, :max_seq_len].to(device)
    labels_all = toks["labels"][:, :max_seq_len].to(device)
    mask_all = toks["response_mask"][:, :max_seq_len].to(device).bool()

    # 5) Optionally compute old_log_probs for grpo_clip / multi-epoch.
    need_old = (loss_type == "grpo_clip") or (epochs_per_rollout_batch > 1)
    old_log_probs_all = None
    if need_old:
        old_chunks = []
        policy.eval()
        with torch.no_grad():
            for start in range(0, rollout_batch_size, micro_batch_size):
                ids = input_ids_all[start : start + micro_batch_size]
                lab = labels_all[start : start + micro_batch_size]
                out = get_response_log_probs(policy, ids, lab, return_token_entropy=False)
                old_chunks.append(out["log_probs"].detach())
        old_log_probs_all = torch.cat(old_chunks, dim=0)
        policy.train()

    # 6) Microbatch loop (optionally multi-epoch).
    optimizer.zero_grad(set_to_none=True)
    sum_loss = 0.0
    sum_entropy = 0.0
    sum_resp_tokens = 0
    micro_count = 0
    last_grad_norm = 0.0

    for _epoch in range(epochs_per_rollout_batch):
        order = list(range(rollout_batch_size))
        rng.shuffle(order)
        order_t = torch.tensor(order, dtype=torch.long, device=device)

        ids_all = input_ids_all.index_select(0, order_t)
        lab_all = labels_all.index_select(0, order_t)
        m_all = mask_all.index_select(0, order_t)
        adv_all = advantages.index_select(0, order_t)
        rr_all = raw_rewards_dev.index_select(0, order_t)
        if old_log_probs_all is not None:
            olp_all = old_log_probs_all.index_select(0, order_t)
        else:
            olp_all = None

        for start in range(0, rollout_batch_size, micro_batch_size):
            end = start + micro_batch_size
            ids = ids_all[start:end]
            lab = lab_all[start:end]
            mask = m_all[start:end]

            out = get_response_log_probs(policy, ids, lab, return_token_entropy=True)
            policy_lp = out["log_probs"]

            adv_mb = adv_all[start:end].unsqueeze(-1).to(policy_lp.dtype)
            rr_mb = rr_all[start:end].unsqueeze(-1).to(policy_lp.dtype)
            olp_mb = olp_all[start:end] if olp_all is not None else None

            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_lp,
                response_mask=mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=rr_mb,
                advantages=adv_mb,
                old_log_probs=olp_mb,
                cliprange=cliprange,
            )

            ent = out["token_entropy"]
            mask_sum = mask.sum().clamp(min=1)
            ent_mean = (ent * mask).sum() / mask_sum
            sum_entropy += float(ent_mean.detach().item())
            sum_resp_tokens += int(mask.sum().detach().item())
            sum_loss += float(loss.detach().item()) * gradient_accumulation_steps
            micro_count += 1

            if micro_count % gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                last_grad_norm = float(grad_norm)

    mean_loss = sum_loss / max(micro_count, 1)
    mean_entropy = sum_entropy / max(micro_count, 1)
    mean_resp_len = sum_resp_tokens / max(rollout_batch_size * epochs_per_rollout_batch, 1)

    metrics = {
        "loss": mean_loss,
        "grad_norm": last_grad_norm,
        "mean_response_entropy": mean_entropy,
        "mean_response_len": mean_resp_len,
        **reward_meta,
    }
    examples = []
    for j in range(min(3, len(flat_prompts))):
        examples.append(
            {
                "prompt": flat_prompts[j],
                "response": flat_responses[j],
                "ground_truth": flat_gts[j],
                "raw_reward": float(raw_rewards[j].item()),
            }
        )
    return metrics, examples


def train(
    *,
    run_name: str,
    seed: int,
    n_grpo_steps: int,
    learning_rate: float,
    advantage_eps: float,
    rollout_batch_size: int,
    group_size: int,
    sampling_temperature: float,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    epochs_per_rollout_batch: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    gpu_memory_utilization: float,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    use_std_normalization: bool,
    cliprange: float,
    max_seq_len: int,
    max_grad_norm: float,
    eval_size: int,
    eval_every: int,
    policy_device: str,
    vllm_device: str,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(PROMPT_TEMPLATE_PATH) as f:
        template = f.read()

    train_records = load_train_records("data/gsm8k/train.jsonl")
    print(f"train records: {len(train_records)}")

    eval_prompts, eval_answers = make_eval_split(GSM8K_TEST, template, eval_size, seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(policy_device)

    llm = init_vllm(
        MODEL_PATH,
        device=vllm_device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    policy.train()

    os.makedirs("results", exist_ok=True)
    train_log_path = f"results/grpo_{run_name}.train.jsonl"
    eval_log_path = f"results/grpo_{run_name}.eval.jsonl"
    rollouts_log_path = f"results/grpo_{run_name}.rollouts.jsonl"
    plot_path = f"results/grpo_{run_name}.eval.png"

    device = torch.device(policy_device)
    rng = random.Random(seed)

    train_log = open(train_log_path, "w")
    eval_log = open(eval_log_path, "w")
    rollouts_log = open(rollouts_log_path, "w")

    def do_eval(step_idx: int):
        out_path = f"results/grpo_{run_name}_eval_{step_idx}.jsonl"
        metrics = run_eval(policy, llm, eval_prompts, eval_answers, out_path)
        eval_log.write(json.dumps({"grpo_step": step_idx, **metrics}) + "\n")
        eval_log.flush()
        print(f"[eval @ step {step_idx}] {metrics}")

    try:
        do_eval(0)

        for step in range(1, n_grpo_steps + 1):
            metrics, examples = grpo_step(
                policy=policy,
                tokenizer=tokenizer,
                llm=llm,
                train_records=train_records,
                template=template,
                rng=rng,
                optimizer=optimizer,
                device=device,
                rollout_batch_size=rollout_batch_size,
                group_size=group_size,
                sampling_temperature=sampling_temperature,
                sampling_min_tokens=sampling_min_tokens,
                sampling_max_tokens=sampling_max_tokens,
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                epochs_per_rollout_batch=epochs_per_rollout_batch,
                loss_type=loss_type,
                use_std_normalization=use_std_normalization,
                advantage_eps=advantage_eps,
                cliprange=cliprange,
                max_seq_len=max_seq_len,
                max_grad_norm=max_grad_norm,
            )
            train_log.write(json.dumps({"grpo_step": step, "lr": learning_rate, **metrics}) + "\n")
            train_log.flush()
            print(
                f"[train @ step {step}] loss={metrics['loss']:.4f} reward_mean={metrics['reward_mean']:.3f} entropy={metrics['mean_response_entropy']:.3f}"
            )

            if step % eval_every == 0 or step == n_grpo_steps:
                do_eval(step)
                for ex in examples:
                    rollouts_log.write(json.dumps({"grpo_step": step, **ex}) + "\n")
                rollouts_log.flush()
    finally:
        train_log.close()
        eval_log.close()
        rollouts_log.close()

    plot_eval_rewards(eval_log_path, plot_path)
    print(f"wrote plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--loss_type",
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    )
    parser.add_argument("--use_std_normalization", action="store_true", default=True)
    parser.add_argument(
        "--no_std_normalization", dest="use_std_normalization", action="store_false"
    )
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=1280)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    args = parser.parse_args()

    train(
        run_name=args.run_name,
        seed=args.seed,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_memory_utilization=args.gpu_memory_utilization,
        loss_type=args.loss_type,
        use_std_normalization=args.use_std_normalization,
        cliprange=args.cliprange,
        max_seq_len=args.max_seq_len,
        max_grad_norm=args.max_grad_norm,
        eval_size=args.eval_size,
        eval_every=args.eval_every,
        policy_device=args.policy_device,
        vllm_device=args.vllm_device,
    )


if __name__ == "__main__":
    main()

"""
Smoke test:
uv run python -m cs336_alignment.grpo_train \
    --run_name smoke --n_grpo_steps 1 \
    --rollout_batch_size 4 --group_size 2 \
    --train_batch_size 4 --gradient_accumulation_steps 2 \
    --eval_size 4 --eval_every 1 \
    --sampling_max_tokens 128 --max_seq_len 384

Full run:
uv run python -m cs336_alignment.grpo_train --run_name gsm8k_v1
"""
