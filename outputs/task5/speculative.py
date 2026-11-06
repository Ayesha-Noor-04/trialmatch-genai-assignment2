
import torch


def _forward(model, token_ids, cache, past_length):
    seq_len = token_ids.shape[1]
    position_ids = torch.arange(
        past_length, past_length + seq_len, device=token_ids.device
    ).unsqueeze(0)
    attention_mask = torch.ones(
        (1, past_length + seq_len), dtype=torch.long, device=token_ids.device
    )
    cache_position = torch.arange(
        past_length, past_length + seq_len, device=token_ids.device
    )
    outputs = model(
        input_ids=token_ids,
        past_key_values=cache,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True
    )
    return outputs.logits, outputs.past_key_values


@torch.inference_mode()
def greedy_cached(model, input_ids, max_new_tokens, eos_token_id=None):
    prompt_len = input_ids.shape[1]
    logits, cache = _forward(model, input_ids, None, 0)
    next_logits = logits[:, -1, :]
    generated = []

    for _ in range(max_new_tokens):
        token = next_logits.argmax(dim=-1, keepdim=True)
        generated.append(token)
        if eos_token_id is not None and token.item() == eos_token_id:
            break
        logits, cache = _forward(
            model, token, cache, prompt_len + len(generated) - 1
        )
        next_logits = logits[:, -1, :]

    return torch.cat(generated, dim=1) if generated else input_ids[:, :0]


@torch.inference_mode()
def speculative_greedy(
    target_model, draft_model, input_ids, max_new_tokens,
    draft_k=4, eos_token_id=None
):
    prompt_len = input_ids.shape[1]
    target_logits, target_cache = _forward(target_model, input_ids, None, 0)
    draft_logits, draft_cache = _forward(draft_model, input_ids, None, 0)
    target_next = target_logits[:, -1, :]
    draft_next = draft_logits[:, -1, :]

    generated = []
    drafted = 0
    accepted = 0
    rejected_rounds = 0

    while len(generated) < max_new_tokens:
        k = min(draft_k, max_new_tokens - len(generated))
        candidates = []

        # Draft a block of candidates.
        for _ in range(k):
            token = draft_next.argmax(dim=-1, keepdim=True)
            candidates.append(token)
            drafted += 1

            if eos_token_id is not None and token.item() == eos_token_id:
                break

            logits, draft_cache = _forward(
                draft_model, token, draft_cache,
                prompt_len + len(generated) + len(candidates) - 1
            )
            draft_next = logits[:, -1, :]

        candidate_ids = torch.cat(candidates, dim=1)
        accepted_this_round = 0
        rejected = False
        stop = False

        # Sequential target verification.
        for i in range(candidate_ids.shape[1]):
            target_token = target_next.argmax(dim=-1, keepdim=True)
            candidate = candidate_ids[:, i:i+1]

            if torch.equal(target_token, candidate):
                generated.append(candidate)
                accepted += 1
                accepted_this_round += 1

                if eos_token_id is not None and candidate.item() == eos_token_id:
                    stop = True
                    break

                logits, target_cache = _forward(
                    target_model, candidate, target_cache,
                    prompt_len + len(generated) - 1
                )
                target_next = logits[:, -1, :]
            else:
                rejected = True
                rejected_rounds += 1

                # Remove unaccepted draft proposals from its cache.
                unaccepted = candidate_ids.shape[1] - accepted_this_round
                if unaccepted:
                    draft_cache.crop(-unaccepted)

                # Insert the target's correction token into both caches.
                generated.append(target_token)

                if eos_token_id is not None and target_token.item() == eos_token_id:
                    stop = True
                    break

                logits, target_cache = _forward(
                    target_model, target_token, target_cache,
                    prompt_len + len(generated) - 1
                )
                target_next = logits[:, -1, :]

                logits, draft_cache = _forward(
                    draft_model, target_token, draft_cache,
                    prompt_len + len(generated) - 1
                )
                draft_next = logits[:, -1, :]
                break

        if stop:
            break

        if not rejected:
            # Stop if the requested output length has already been reached.
            if len(generated) >= max_new_tokens:
                break

            # All proposals matched; append the target bonus token.
            bonus = target_next.argmax(dim=-1, keepdim=True)
            generated.append(bonus)

            if eos_token_id is not None and bonus.item() == eos_token_id:
                break

            logits, target_cache = _forward(
                target_model, bonus, target_cache,
                prompt_len + len(generated) - 1
            )
            target_next = logits[:, -1, :]

            logits, draft_cache = _forward(
                draft_model, bonus, draft_cache,
                prompt_len + len(generated) - 1
            )
            draft_next = logits[:, -1, :]

    output = torch.cat(generated, dim=1) if generated else input_ids[:, :0]
    stats = {
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "rejected_rounds": rejected_rounds
    }
    return output, stats
