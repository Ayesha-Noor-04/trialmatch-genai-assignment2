
import time
import torch
from transformers.cache_utils import DynamicCache

@torch.inference_mode()
def greedy_no_cache(model, input_ids, max_new_tokens):
    """Reference decoding: recompute the complete sequence every step."""
    sequence = input_ids.clone()
    device = sequence.device
    prefill_time = 0.0
    decode_time = 0.0

    for step in range(max_new_tokens):
        positions = torch.arange(
            sequence.shape[1], device=device
        ).unsqueeze(0)

        attention_mask = torch.ones_like(sequence)

        if step == 0:
            torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(
                input_ids=sequence,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False
            )

            torch.cuda.synchronize()
            prefill_time = time.perf_counter() - start
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(
                input_ids=sequence,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False
            )

            torch.cuda.synchronize()
            decode_time += time.perf_counter() - start

        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        sequence = torch.cat([sequence, next_token], dim=-1)

    return sequence[:, input_ids.shape[1]:], prefill_time, decode_time


@torch.inference_mode()
def greedy_with_cache(model, input_ids, max_new_tokens):
    """Cached decoding: prefill once, then process one token per step."""
    device = input_ids.device
    prompt_length = input_ids.shape[1]
    sequence = input_ids.clone()

    cache = DynamicCache(config=model.config)

    # Prefill: process the complete prompt once.
    positions = torch.arange(
        prompt_length, device=device
    ).unsqueeze(0)

    attention_mask = torch.ones_like(input_ids)
    cache_position = torch.arange(prompt_length, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True
    )

    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - start

    cache = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1, keepdim=True
    )

    generated = [next_token]
    decode_time = 0.0

    # Decode: feed only the latest token and update the cache.
    for step in range(1, max_new_tokens):
        past_length = cache.get_seq_length()

        position_ids = torch.tensor(
            [[past_length]], device=device
        )
        cache_position = torch.tensor(
            [past_length], device=device
        )

        # Mask covers the prompt and all previously generated tokens.
        full_attention_mask = torch.ones(
            (1, past_length + 1),
            dtype=torch.long,
            device=device
        )

        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = model(
            input_ids=next_token,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True
        )

        torch.cuda.synchronize()
        decode_time += time.perf_counter() - start

        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        generated.append(next_token)

    return torch.cat(generated, dim=-1), prefill_time, decode_time
