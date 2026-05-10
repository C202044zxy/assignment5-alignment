from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize prompts and outputs, returning input_ids, labels, and a response mask.

    Args:
        prompt_strs: list[str], prompt strings.
        output_strs: list[str], output strings.
        tokenizer: PreTrainedTokenizer.

    Returns:
        dict with keys:
            input_ids:     (batch_size, max_len - 1) tokens with last position sliced off
            labels:        (batch_size, max_len - 1) tokens shifted left by one
            response_mask: (batch_size, max_len - 1) 1 where labels are response tokens, else 0
    """
    assert len(prompt_strs) == len(output_strs)
    batch_size = len(prompt_strs)

    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    max_len = max([len(p) + len(o) for p, o in zip(prompt_ids, output_ids)])
    
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    
    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    response_mask_full = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        seq = p + o
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        response_mask_full[i, len(p): len(seq)] = 1

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]
    response_mask = response_mask_full[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    lse = torch.logsumexp(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    return lse - torch.sum(logits * p, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    
    logits = model(input_ids=input_ids).logits  # (batch, seq_len, vocab_size)
    
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    
    return result


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "/data/models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen2.5-Math-1.5B")
