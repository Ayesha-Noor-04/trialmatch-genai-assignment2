
import math
import torch


def reference_attention(q, k, v, causal=True):
    """
    Reference scaled dot-product attention.
    Inputs: [batch, heads, sequence, head_dim]
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if causal:
        q_len, k_len = q.shape[-2], k.shape[-2]
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
        mask = q_pos >= k_pos
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def tiled_online_attention(q, k, v, causal=True, block_size=16):
    """
    Tiled attention using numerically stable online softmax.
    Does not materialize the complete attention score matrix.
    Inputs: [batch, heads, sequence, head_dim]
    """
    batch, heads, q_len, d = q.shape
    k_len = k.shape[-2]

    output = torch.empty_like(q)

    for q_start in range(0, q_len, block_size):
        q_end = min(q_start + block_size, q_len)
        q_block = q[:, :, q_start:q_end, :]
        rows = q_end - q_start

        running_max = torch.full(
            (batch, heads, rows, 1),
            float("-inf"), device=q.device, dtype=torch.float32
        )
        running_sum = torch.zeros(
            (batch, heads, rows, 1),
            device=q.device, dtype=torch.float32
        )
        running_value = torch.zeros(
            (batch, heads, rows, d),
            device=q.device, dtype=torch.float32
        )

        for k_start in range(0, k_len, block_size):
            k_end = min(k_start + block_size, k_len)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            scores = torch.matmul(
                q_block.float(), k_block.float().transpose(-2, -1)
            ) / math.sqrt(d)

            if causal:
                q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                k_pos = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                mask = q_pos >= k_pos
                scores = scores.masked_fill(~mask, float("-inf"))

            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, block_max)

            old_scale = torch.exp(running_max - new_max)
            old_scale = torch.nan_to_num(old_scale, nan=0.0)

            probabilities = torch.exp(scores - new_max)
            probabilities = torch.nan_to_num(probabilities, nan=0.0)

            running_sum = (
                old_scale * running_sum
                + probabilities.sum(dim=-1, keepdim=True)
            )

            running_value = (
                old_scale * running_value
                + torch.matmul(probabilities, v_block.float())
            )

            running_max = new_max

        result = running_value / running_sum
        output[:, :, q_start:q_end, :] = result.to(output.dtype)

    return output
