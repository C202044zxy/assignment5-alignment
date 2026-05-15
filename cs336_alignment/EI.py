import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_train import (
    GSM8K_TEST,
    GSM8K_TRAIN,
    MODEL_PATH,
    PROMPT_TEMPLATE_PATH,
    format_gsm8k_example,
    init_vllm,
    load_policy_into_vllm_instance,
    make_eval_split,
    train,
)


def load_train_records(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            gold = r["answer"].split("####")[1].strip()
            records.append({"question": r["question"], "gold": gold, "raw": r})
    return records


def rollout(
    llm: LLM,
    prompts: List[str],
    G: int,
    temperature: float,
    max_tokens: int,
) -> List[List[str]]:
    sampling = SamplingParams(
        n=G,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling)
    return [[c.text for c in o.outputs] for o in outputs]


def build_sft_dataset(
    prompts: List[str],
    gold_answers: List[str],
    rollouts: List[List[str]],
) -> Tuple[List[dict], int, int]:
    kept = []
    n_total = 0
    for prompt, gold, responses in zip(prompts, gold_answers, rollouts):
        for response in responses:
            n_total += 1
            r = r1_zero_reward_fn(response, gold, fast=True)
            if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0:
                kept.append({"prompt": prompt, "response": response, "answer": gold})
    return kept, n_total, len(kept)


def read_train_log_entropy(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    vals = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "mean_response_entropy" in rec:
                vals.append(rec["mean_response_entropy"])
    return float(sum(vals) / len(vals)) if vals else None


def read_eval_log_accuracies(path: str) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.exists(path):
        return None, None
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return None, None
    pre = records[0].get("avg_answer_reward")
    post = records[-1].get("avg_answer_reward")
    return pre, post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--Db", type=int, default=1024)
    parser.add_argument("--epochs_per_ei", type=int, default=2)
    parser.add_argument("--rollout_temperature", type=float, default=1.0)
    parser.add_argument("--rollout_max_tokens", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_rollout", action="store_true",
                        help="Smoke-test: use gold GSM8K examples instead of vLLM rollouts.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(PROMPT_TEMPLATE_PATH) as f:
        template = f.read()

    train_records = load_train_records(GSM8K_TRAIN)
    print(f"train records: {len(train_records)}")

    eval_prompts, eval_answers = make_eval_split(
        GSM8K_TEST, template, args.eval_size, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.policy_device)

    need_vllm = not (args.skip_rollout and args.skip_eval)
    if need_vllm:
        llm = init_vllm(
            MODEL_PATH,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    else:
        llm = None

    os.makedirs("results", exist_ok=True)
    ei_log_path = f"results/ei_{args.run_name}.ei.jsonl"
    device = torch.device(args.policy_device)

    with open(ei_log_path, "w") as ei_log:
        for i in range(args.n_ei_steps):
            rng = random.Random(args.seed + i)
            sampled = rng.sample(train_records, min(args.Db, len(train_records)))
            prompts = [template.format(question=r["question"]) for r in sampled]
            gold_answers = [r["gold"] for r in sampled]

            if args.skip_rollout:
                kept = [
                    {
                        "prompt": template.format(question=r["raw"]["question"]),
                        "response": format_gsm8k_example(r["raw"], template)["response"],
                        "answer": r["gold"],
                    }
                    for r in sampled
                ]
                n_rollouts = len(kept)
                n_kept = len(kept)
            else:
                load_policy_into_vllm_instance(policy, llm)
                rollouts = rollout(
                    llm,
                    prompts,
                    G=args.G,
                    temperature=args.rollout_temperature,
                    max_tokens=args.rollout_max_tokens,
                )
                kept, n_rollouts, n_kept = build_sft_dataset(prompts, gold_answers, rollouts)

            kept_frac = (n_kept / n_rollouts) if n_rollouts > 0 else 0.0
            print(f"[ei {i}] rollouts={n_rollouts} kept={n_kept} kept_frac={kept_frac:.3f}")

            step_run = f"{args.run_name}_step{i}"
            train_log_path = f"results/sft_{step_run}.train.jsonl"
            eval_log_path = f"results/sft_{step_run}.eval.jsonl"

            if n_kept == 0:
                print(f"[ei {i}] no correct rollouts — skipping SFT")
                ei_log.write(json.dumps({
                    "ei_step": i,
                    "n_rollouts": n_rollouts,
                    "n_kept": n_kept,
                    "kept_frac": kept_frac,
                    "eval_accuracy_pre": None,
                    "eval_accuracy_post": None,
                    "mean_train_entropy": None,
                }) + "\n")
                ei_log.flush()
                continue

            train(
                policy=policy,
                llm=None if args.skip_eval else llm,
                tokenizer=tokenizer,
                train_examples=kept,
                eval_prompts=eval_prompts,
                eval_answers=eval_answers,
                lr=args.lr,
                batch_size=args.micro_batch_size,
                gradient_accumulation_steps=args.grad_accum,
                epochs=args.epochs_per_ei,
                eval_every=10**9,
                run_name=step_run,
                device=device,
                max_seq_len=args.max_seq_len,
            )

            pre_acc, post_acc = read_eval_log_accuracies(eval_log_path)
            mean_ent = read_train_log_entropy(train_log_path)

            ei_log.write(json.dumps({
                "ei_step": i,
                "n_rollouts": n_rollouts,
                "n_kept": n_kept,
                "kept_frac": kept_frac,
                "eval_accuracy_pre": pre_acc,
                "eval_accuracy_post": post_acc,
                "mean_train_entropy": mean_ent,
            }) + "\n")
            ei_log.flush()
            print(f"[ei {i}] pre={pre_acc} post={post_acc} mean_entropy={mean_ent}")


if __name__ == "__main__":
    main()

"""
uv run python -m cs336_alignment.EI \
    --run_name smoke --n_ei_steps 1 --G 2 --Db 4 --epochs_per_ei 1 \
    --micro_batch_size 2 --grad_accum 1 --max_seq_len 256 \
    --skip_eval --skip_rollout
"""
