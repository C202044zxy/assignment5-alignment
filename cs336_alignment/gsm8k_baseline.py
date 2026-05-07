from vllm import LLM, SamplingParams
from typing import Callable, List
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import os


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    out_path = "results/gsm8k_baseline.jsonl",
):
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    sum_format, sum_answer = 0.0, 0.0
    with open(out_path, "w") as f:
        for prompt, output, answer in zip(prompts, outputs, answers):
            response = output.outputs[0].text
            reward = reward_fn(response, answer)
            sum_format += reward["format_reward"]
            sum_answer += reward["answer_reward"]
            f.write(json.dumps({
                "prompt": prompt,
                "response": response,
                "answer": answer,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
            }) + "\n")

    n = len(prompts)
    metrics = {
        "avg_format_reward": sum_format / n,
        "avg_answer_reward": sum_answer / n, 
    }
    with open(out_path.replace(".jsonl", ".metrics.jsonl"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        template = f.read()

    prompts: List[str] = []
    answers: List[str] = []
    with open("data/gsm8k/test.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            prompts.append(template.format(question=record["question"]))
            answers.append(record["answer"].split("####")[1].strip())

    llm = LLM(model="data/models/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        eval_sampling_params=sampling_params,
    )

    print(metrics)