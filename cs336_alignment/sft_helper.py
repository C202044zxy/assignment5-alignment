from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
import torch.nn.functional as F
from typing import List


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
):
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
    """
    Compute the entropy of the categorical distribution defined by `logits`.

    Uses the numerically stable form H = logsumexp(z) - sum(softmax(z) * z),
    which avoids materializing log-probs explicitly.

    Args:
        logits: (..., vocab_size) unnormalized logits.

    Returns:
        Tensor of shape (...,) with the per-position entropy in nats.
    """
    lse = torch.logsumexp(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    return lse - torch.sum(logits * p, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """
    Run `model` on `input_ids` and gather the log-probability of each `labels` token.

    Args:
        model: causal LM that returns logits of shape (batch, seq_len, vocab_size).
        input_ids: (batch, seq_len) token ids fed to the model.
        labels: (batch, seq_len) target token ids to score.
        return_token_entropy: if True, also include per-token entropy of the predicted distribution.

    Returns:
        dict with:
            log_probs:     (batch, seq_len) log-prob of each label token under the model.
            token_entropy: (batch, seq_len) entropy of the predicted distribution (only if requested).
    """
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


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum `tensor` along `dim` over entries selected by `mask`, then divide by `normalize_constant`.

    Args:
        tensor: values to aggregate.
        mask: boolean tensor broadcastable to `tensor`; True keeps the entry, False zeros it out.
        normalize_constant: scalar divisor applied after summation.
        dim: axis to reduce over; if None, sums over all dimensions.

    Returns:
        Tensor of the masked, normalized sum.
    """
    masked = tensor.masked_fill(~mask, 0)
    return torch.sum(masked, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Run one SFT microbatch: compute the response-masked NLL loss, scale it for gradient accumulation, and backpropagate.

    The returned loss is already divided by `batch_size * gradient_accumulation_steps`, so calling
    `optimizer.step()` after `gradient_accumulation_steps` microbatches yields the correct mean-update.

    Args:
        policy_log_probs: (batch, seq_len) log-probs from the policy for each label token.
        response_mask: (batch, seq_len) 1 on response tokens that should contribute to the loss.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: divisor applied inside `masked_normalize` (e.g. number of response tokens).

    Returns:
        (loss, metadata) where `loss` is the scaled scalar loss and `metadata` contains the detached loss.
    """
    batch_size = policy_log_probs.shape[0]

    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    loss = loss / (batch_size * gradient_accumulation_steps)
    loss.backward()

    metadata = {"loss": loss.detach()}

    return (loss, metadata)


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "/data/models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen2.5-Math-1.5B")
