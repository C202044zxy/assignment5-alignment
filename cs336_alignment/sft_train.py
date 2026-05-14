import argparse
import json
import os
import random
import re
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.gsm8k_baseline import evaluate_vllm
from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


MODEL_PATH = "data/models/Qwen2.5-Math-1.5B"
PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"
GSM8K_TRAIN = "data/gsm8k/train.jsonl"
GSM8K_TEST = "data/gsm8k/test.jsonl"
CALC_PATTERN = re.compile(r"<<[^>]*>>")


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def format_gsm8k_example(record: dict, template: str) -> dict:
    reasoning, _, final = record["answer"].partition("####")
    reasoning = CALC_PATTERN.sub("", reasoning).strip()
    final = final.strip()
    return {
        "prompt": template.format(question=record["question"]),
        "response": f" {reasoning} </think> <answer> {final} </answer>",
        "answer": final,
    }


def load_sft_examples(path: str, template: str) -> List[dict]:
    with open(path) as f:
        return [format_gsm8k_example(json.loads(line), template) for line in f]


def filter_correct(examples: List[dict]) -> List[dict]:
    kept = []
    for ex in examples:
        r = r1_zero_reward_fn(ex["response"], ex["answer"], fast=True)
        if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0:
            kept.append(ex)
    return kept


def make_eval_split(path: str, template: str, eval_size: int, seed: int):
    prompts, answers = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(template.format(question=r["question"]))
            answers.append(r["answer"].split("####")[1].strip())
    idx = list(range(len(prompts)))
    random.Random(seed).shuffle(idx)
    idx = idx[:eval_size]
    return [prompts[i] for i in idx], [answers[i] for i in idx]


def run_eval(
    policy: PreTrainedModel,
    llm: LLM,
    prompts: List[str],
    answers: List[str],
    out_path: str,
) -> dict:
    policy.eval()
    load_policy_into_vllm_instance(policy, llm)
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    with torch.no_grad():
        metrics = evaluate_vllm(
            vllm_model=llm,
            reward_fn=r1_zero_reward_fn,
            prompts=prompts,
            answers=answers,
            eval_sampling_params=sampling,
            out_path=out_path,
        )
    policy.train()
    return metrics


def train(
    policy: PreTrainedModel,
    llm: Optional[LLM],
    tokenizer: PreTrainedTokenizer,
    train_examples: List[dict],
    eval_prompts: List[str],
    eval_answers: List[str],
    *,
    lr: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    eval_every: int,
    run_name: str,
    device: torch.device,
    max_seq_len: int,
):
    os.makedirs("results", exist_ok=True)
    train_log_path = f"results/sft_{run_name}.train.jsonl"
    eval_log_path = f"results/sft_{run_name}.eval.jsonl"

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    policy.train()

    n = len(train_examples)
    train_step = 0
    micro_step = 0
    eval_step = 0

    with open(train_log_path, "w") as train_log, open(eval_log_path, "w") as eval_log:

        def do_eval():
            nonlocal eval_step
            if llm is None:
                print(f"[eval {eval_step}] skipped (no vLLM)")
                eval_step += 1
                return
            out = f"results/sft_{run_name}_eval_{eval_step}.jsonl"
            metrics = run_eval(policy, llm, eval_prompts, eval_answers, out)
            eval_log.write(json.dumps({
                "eval_step": eval_step,
                "train_step": train_step,
                **metrics,
            }) + "\n")
            eval_log.flush()
            print(f"[eval {eval_step}] train_step={train_step} {metrics}")
            eval_step += 1

        do_eval()

        for epoch in range(epochs):
            order = list(range(n))
            random.shuffle(order)
            for start in range(0, n, batch_size):
                batch = [train_examples[i] for i in order[start:start + batch_size]]
                toks = tokenize_prompt_and_output(
                    [b["prompt"] for b in batch],
                    [b["response"] for b in batch],
                    tokenizer,
                )
                input_ids = toks["input_ids"][:, :max_seq_len].to(device)
                labels = toks["labels"][:, :max_seq_len].to(device)
                response_mask = toks["response_mask"][:, :max_seq_len].to(device).bool()

                out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=out["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    normalize_constant=1.0,
                )

                micro_step += 1
                if micro_step % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_step += 1

                    train_log.write(json.dumps({
                        "train_step": train_step,
                        "epoch": epoch,
                        "loss": float(loss.detach().item()) * gradient_accumulation_steps,
                        "grad_norm": float(grad_norm),
                        "lr": lr,
                    }) + "\n")
                    train_log.flush()

                    if train_step % eval_every == 0:
                        do_eval()

        if micro_step % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        do_eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=str, default="full",
                        help="int or 'full'")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--filter_correct", action="store_true")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip vLLM eval (for smoke-testing on a single small GPU)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(PROMPT_TEMPLATE_PATH) as f:
        template = f.read()

    examples = load_sft_examples(GSM8K_TRAIN, template)
    if args.filter_correct:
        before = len(examples)
        examples = filter_correct(examples)
        print(f"filtered: {before} -> {len(examples)}")

    random.Random(args.seed).shuffle(examples)
    if args.n_examples != "full":
        examples = examples[: int(args.n_examples)]
    print(f"training examples: {len(examples)}")

    eval_prompts, eval_answers = make_eval_split(
        GSM8K_TEST, template, args.eval_size, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.policy_device)

    if args.skip_eval:
        llm = None
    else:
        llm = init_vllm(
            MODEL_PATH,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )

    train(
        policy=policy,
        llm=llm,
        tokenizer=tokenizer,
        train_examples=examples,
        eval_prompts=eval_prompts,
        eval_answers=eval_answers,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        eval_every=args.eval_every,
        run_name=args.run_name,
        device=torch.device(args.policy_device),
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()

"""
uv run python -m cs336_alignment.sft_train \
    --n_examples 8 --epochs 1 \
    --eval_every 2 --eval_size 4 \
    --run_name smoke \
    --batch_size 2 --gradient_accumulation_steps 1 \
    --max_seq_len 256 \
    --skip_eval
"""
