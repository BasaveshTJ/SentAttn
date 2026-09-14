import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from sentence import split_chat_sentences, split_sentences


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_NEW_TOKENS = 100
TOP_N = 2


def sample_next_token(logits, generated_ids, temperature, top_p, repetition_penalty):
    scores = logits.clone() / max(1e-5, float(temperature))
    if repetition_penalty > 1.0:
        for tid in set(generated_ids):
            if 0 <= tid < scores.shape[-1]:
                scores[tid] = scores[tid] / repetition_penalty if scores[tid] > 0 else scores[tid] * repetition_penalty
    probs = F.softmax(scores, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cutoff = torch.cumsum(sorted_probs, dim=-1) > top_p
        if cutoff.any():
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / torch.clamp_min(sorted_probs.sum(), 1e-8)
            return int(sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item())
    return int(torch.multinomial(probs, num_samples=1).item())


def pool_kv(k, v, mode):
    if mode == "mean":
        return k.mean(dim=2, keepdim=True), v.mean(dim=2, keepdim=True)
    if mode == "max":
        return k.max(dim=2, keepdim=True).values, v.max(dim=2, keepdim=True).values
    raise ValueError(f"Unknown pooling mode: {mode}")


def split_kv(kv_cat):
    dim = kv_cat.shape[-1] // 2
    return kv_cat[..., :dim], kv_cat[..., dim:]


def find_subsequence(haystack, needle, start):
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    for i in range(start, max(start, limit)):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def tokenize_prompt_with_spans(tokenizer, prompt):
    if isinstance(prompt, list):
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0].tolist()
        chunks = [seg.split(": ", 1)[1] if ": " in seg else seg for seg in split_chat_sentences(prompt)]
        segments = split_chat_sentences(prompt)
    else:
        prompt_text = prompt
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        chunks = split_sentences(prompt)
        segments = chunks

    spans = []
    cursor = 0
    for chunk in chunks:
        found, found_len = -1, 0
        for variant in (chunk, " " + chunk, "\n" + chunk):
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            pos = find_subsequence(prompt_ids, ids, cursor)
            if pos != -1:
                found, found_len = pos, len(ids)
                break
        if found == -1:
            continue
        spans.append((found, found + found_len))
        cursor = found + found_len

    return prompt_text, prompt_ids, spans, segments


def init_memory_from_prompt(past_key_values, spans, pool_mode):
    pooled_by_layer = [[] for _ in range(len(past_key_values))]
    full_by_layer = [[] for _ in range(len(past_key_values))]
    for layer_idx, layer_past in enumerate(past_key_values):
        k, v = layer_past[0], layer_past[1]
        for start, end in spans:
            seg_k = k[:, :, start:end, :]
            seg_v = v[:, :, start:end, :]
            pk, pv = pool_kv(seg_k, seg_v, pool_mode)
            pooled_by_layer[layer_idx].append(torch.cat([pk, pv], dim=-1))
            full_by_layer[layer_idx].append(torch.cat([seg_k, seg_v], dim=-1))
    return pooled_by_layer, full_by_layer


def build_cache(pooled_by_layer, full_by_layer, selected_ids, always_full_ids, active_k, active_v, model):
    cache = DynamicCache(config=model.config)
    for layer_idx in range(len(pooled_by_layer)):
        parts_k, parts_v = [], []
        full_set = set(always_full_ids) | set(selected_ids)

        for sent_idx, pooled_cat in enumerate(pooled_by_layer[layer_idx]):
            if sent_idx in full_set:
                continue
            pk, pv = split_kv(pooled_cat)
            parts_k.append(pk)
            parts_v.append(pv)

        for sent_idx in sorted(full_set):
            if 0 <= sent_idx < len(full_by_layer[layer_idx]):
                fk, fv = split_kv(full_by_layer[layer_idx][sent_idx])
                parts_k.append(fk)
                parts_v.append(fv)

        if active_k[layer_idx] is not None:
            parts_k.append(active_k[layer_idx])
            parts_v.append(active_v[layer_idx])

        if parts_k:
            cache.update(torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2), layer_idx)
    return cache


def select_top_sentences(estimate_out, pooled_by_layer, top_n):
    if top_n <= 0 or not pooled_by_layer or not pooled_by_layer[0]:
        return [], []

    sentence_count = len(pooled_by_layer[0])
    scores = torch.zeros(sentence_count, dtype=torch.float32)

    for layer_idx, layer_past in enumerate(estimate_out.past_key_values):
        query_k = layer_past[0][:, :, -1:, :].reshape(-1).float()
        query_norm = torch.norm(query_k) + 1e-8
        for sent_idx in range(sentence_count):
            pooled_k, _ = split_kv(pooled_by_layer[layer_idx][sent_idx])
            pooled_vec = pooled_k.reshape(-1).float()
            pooled_norm = torch.norm(pooled_vec) + 1e-8
            scores[sent_idx] += torch.dot(query_k, pooled_vec) / (query_norm * pooled_norm)

    ranked = torch.argsort(scores, descending=True).tolist()
    selected = ranked[: min(top_n, len(ranked))]
    return selected, scores.tolist()


def append_active_token(final_past, active_k, active_v):
    for layer_idx, layer_past in enumerate(final_past):
        k, v = layer_past[0][:, :, -1:, :], layer_past[1][:, :, -1:, :]
        active_k[layer_idx] = k if active_k[layer_idx] is None else torch.cat([active_k[layer_idx], k], dim=2)
        active_v[layer_idx] = v if active_v[layer_idx] is None else torch.cat([active_v[layer_idx], v], dim=2)


def close_active_sentence(active_k, active_v, pooled_by_layer, full_by_layer, pool_mode):
    for layer_idx in range(len(pooled_by_layer)):
        if active_k[layer_idx] is None:
            continue
        pk, pv = pool_kv(active_k[layer_idx], active_v[layer_idx], pool_mode)
        pooled_by_layer[layer_idx].append(torch.cat([pk, pv], dim=-1))
        full_by_layer[layer_idx].append(torch.cat([active_k[layer_idx], active_v[layer_idx]], dim=-1))
        active_k[layer_idx], active_v[layer_idx] = None, None


def run_sentence_representation_inference(prompt, pool_mode, max_new_tokens, top_n, keep_last_full_prompt, temperature, top_p, repetition_penalty, force_close_every):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device).eval()

    prompt_text, prompt_ids, spans, segments = tokenize_prompt_with_spans(tokenizer, prompt)
    if not prompt_ids or not spans:
        raise ValueError("No valid sentence spans produced from prompt")

    with torch.no_grad():
        prompt_out = model(input_ids=torch.tensor([prompt_ids], device=device), use_cache=True)

    pooled_by_layer, full_by_layer = init_memory_from_prompt(prompt_out.past_key_values, spans, pool_mode)
    always_full_ids = list(range(max(0, len(spans) - max(0, keep_last_full_prompt)), len(spans)))

    layer_count = model.config.num_hidden_layers
    active_k = [None] * layer_count
    active_v = [None] * layer_count
    generated = []
    selected_snapshot, score_snapshot = [], []
    next_id = sample_next_token(prompt_out.logits[0, -1, :], [], temperature, top_p, repetition_penalty)
    tokens_since_close = 0
    close_events = 0

    for _ in range(max_new_tokens):
        token_tensor = torch.tensor([[next_id]], device=device)
        generated.append(next_id)
        absolute_pos = len(prompt_ids) + len(generated) - 1
        position_ids = torch.tensor([[absolute_pos]], device=device)
        cache_position = torch.tensor([absolute_pos], device=device)

        estimate_cache = build_cache(
            pooled_by_layer,
            full_by_layer,
            selected_ids=[],
            always_full_ids=always_full_ids,
            active_k=active_k,
            active_v=active_v,
            model=model,
        )
        with torch.no_grad():
            estimate_out = model(
                input_ids=token_tensor,
                past_key_values=estimate_cache,
                use_cache=True,
                position_ids=position_ids,
                cache_position=cache_position,
            )

        selected, scores = select_top_sentences(estimate_out, pooled_by_layer, top_n)
        selected_snapshot, score_snapshot = selected, scores

        final_cache = build_cache(
            pooled_by_layer,
            full_by_layer,
            selected_ids=selected,
            always_full_ids=always_full_ids,
            active_k=active_k,
            active_v=active_v,
            model=model,
        )
        with torch.no_grad():
            final_out = model(
                input_ids=token_tensor,
                past_key_values=final_cache,
                use_cache=True,
                position_ids=position_ids,
                cache_position=cache_position,
            )

        append_active_token(final_out.past_key_values, active_k, active_v)
        next_id = sample_next_token(
            final_out.logits[0, -1, :],
            generated,
            temperature,
            top_p,
            repetition_penalty,
        )

        tokens_since_close += 1
        piece = tokenizer.decode([generated[-1]], skip_special_tokens=False)
        force_close = force_close_every > 0 and tokens_since_close >= force_close_every
        if any(p in piece for p in (".", "!", "?")) or force_close:
            close_active_sentence(active_k, active_v, pooled_by_layer, full_by_layer, pool_mode)
            tokens_since_close = 0
            close_events += 1

    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    active_tokens = int(active_k[0].shape[2]) if active_k and active_k[0] is not None else 0
    completed_slots = len(pooled_by_layer[0]) if pooled_by_layer else 0
    estimated_kv_slots = int(layer_count * (completed_slots + active_tokens + len(selected_snapshot)))

    return {
        "pool_mode": pool_mode,
        "top_n": top_n,
        "segments": segments,
        "completed_sentence_slots": completed_slots,
        "active_sentence_tokens": active_tokens,
        "sentence_close_events": close_events,
        "generated_tokens": len(generated),
        "generated_text": generated_text,
        "full_text": prompt_text + generated_text,
        "estimated_kv_slots": estimated_kv_slots,
        "last_step_selected_sentence_ids": selected_snapshot,
        "last_step_global_sentence_scores": score_snapshot,
    }


def main():
    parser = argparse.ArgumentParser(description="Sentence pooled + selective full retrieval inference")
    parser.add_argument("--pool", choices=["mean", "max"], default="mean")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--keep-last-full-prompt", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--force-close-every", type=int, default=24)
    args = parser.parse_args()

    prompt = [
        {"content": "Hi there", "role": "user"},
        {"content": "Hello! How can I help you today?", "role": "assistant"},
        {"content": "I'm looking for a beach resort for my next vacation. Can you recommend some popular ones?", "role": "user"},
    ]

    result = run_sentence_representation_inference(
        prompt,
        pool_mode=args.pool,
        max_new_tokens=args.max_new_tokens,
        top_n=args.top_n,
        keep_last_full_prompt=args.keep_last_full_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        force_close_every=args.force_close_every,
    )

    print("=== Sentence Representation Retrieval Inference ===")
    print("Pool mode:", result["pool_mode"])
    print("Top N retrieved sentences/layer:", result["top_n"])
    print("Sentence segments:", result["segments"])
    print("Completed sentence slots:", result["completed_sentence_slots"])
    print("Active sentence token count:", result["active_sentence_tokens"])
    print("Sentence close events:", result["sentence_close_events"])
    print("Generated tokens:", result["generated_tokens"])
    print("Estimated KV slots kept:", result["estimated_kv_slots"])
    print("Last-step selected sentence ids:", result["last_step_selected_sentence_ids"])
    print("Last-step global sentence scores:", result["last_step_global_sentence_scores"])
    print("Generated text:", result["generated_text"])
    print("Full text:", result["full_text"])


if __name__ == "__main__":
    main()
