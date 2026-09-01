import argparse
from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from sentence import split_chat_sentences, split_sentences


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_NEW_TOKENS = 100
TOP_N = 2


def _sample_next_token(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> int:
    next_logits = logits.clone()

    if repetition_penalty > 1.0 and generated_ids:
        seen = set(generated_ids)
        for tid in seen:
            if 0 <= tid < next_logits.shape[-1]:
                val = next_logits[tid]
                if val > 0:
                    next_logits[tid] = val / repetition_penalty
                else:
                    next_logits[tid] = val * repetition_penalty

    temp = max(1e-5, float(temperature))
    next_logits = next_logits / temp

    probs = F.softmax(next_logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        if mask.any():
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            denom = sorted_probs.sum()
            if denom.item() > 0:
                sorted_probs = sorted_probs / denom
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[sampled].item())

    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


def _pool_kv(k: torch.Tensor, v: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    # k/v shape: [batch, heads, seq, head_dim]
    if mode == "mean":
        return k.mean(dim=2, keepdim=True), v.mean(dim=2, keepdim=True)
    if mode == "max":
        return k.max(dim=2, keepdim=True).values, v.max(dim=2, keepdim=True).values
    raise ValueError(f"Unknown pooling mode: {mode}")


def _find_subsequence(haystack: List[int], needle: List[int], start_idx: int) -> int:
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    if limit <= start_idx:
        return -1
    for i in range(start_idx, limit):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _tokenize_prompt_with_spans(tokenizer, prompt: str | Sequence[dict]) -> tuple[str, List[int], List[tuple[int, int]], List[str]]:
    if isinstance(prompt, list):
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_tok = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(prompt_tok, torch.Tensor):
            prompt_ids = prompt_tok[0].tolist() if prompt_tok.dim() > 1 else prompt_tok.tolist()
        elif hasattr(prompt_tok, "input_ids"):
            ids = prompt_tok.input_ids
            if isinstance(ids, torch.Tensor):
                prompt_ids = ids[0].tolist() if ids.dim() > 1 else ids.tolist()
            else:
                prompt_ids = list(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else list(ids)
        elif isinstance(prompt_tok, list):
            prompt_ids = list(prompt_tok)
        elif hasattr(prompt_tok, "ids"):
            prompt_ids = list(prompt_tok.ids)
        else:
            raise TypeError(f"Unsupported chat-template tokenized output type: {type(prompt_tok)}")

        chat_segments = split_chat_sentences(prompt)
        chunks: List[str] = []
        for seg in chat_segments:
            if ": " in seg:
                chunks.append(seg.split(": ", 1)[1])
            else:
                chunks.append(seg)

        spans: List[tuple[int, int]] = []
        cursor = 0
        for chunk in chunks:
            variants = [chunk, " " + chunk, "\n" + chunk]
            found_start = -1
            found_len = 0
            for v in variants:
                ids = tokenizer(v, add_special_tokens=False).input_ids
                pos = _find_subsequence(prompt_ids, ids, cursor)
                if pos != -1:
                    found_start = pos
                    found_len = len(ids)
                    break
            if found_start == -1:
                raise ValueError(f"Could not map sentence chunk into chat-template tokens: {chunk!r}")
            spans.append((found_start, found_start + found_len))
            cursor = found_start + found_len
        return prompt_text, prompt_ids, spans, chat_segments

    segments = split_sentences(prompt)
    pieces: List[str] = []
    spans: List[tuple[int, int]] = []
    prompt_ids: List[int] = []
    for i, seg in enumerate(segments):
        clean = seg.strip()
        if not clean:
            continue
        piece = clean if i == 0 else " " + clean
        ids = tokenizer(piece, add_special_tokens=False).input_ids
        if not ids:
            continue
        start = len(prompt_ids)
        prompt_ids.extend(ids)
        spans.append((start, len(prompt_ids)))
        pieces.append(piece)
    prompt_text = "".join(pieces)
    return prompt_text, prompt_ids, spans, segments


def _init_sentence_memory_from_prompt(past_key_values, spans: Sequence[tuple[int, int]], pool_mode: str):
    num_layers = len(past_key_values)
    pooled_by_layer: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    full_by_layer: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

    for li, layer_past in enumerate(past_key_values):
        k, v = layer_past[0], layer_past[1]
        for start, end in spans:
            seg_k = k[:, :, start:end, :]
            seg_v = v[:, :, start:end, :]
            pk, pv = _pool_kv(seg_k, seg_v, pool_mode)
            pooled_by_layer[li].append(torch.cat([pk, pv], dim=-1))
            full_by_layer[li].append(torch.cat([seg_k, seg_v], dim=-1))

    return pooled_by_layer, full_by_layer


def _split_kv(kv_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = kv_cat.shape[-1] // 2
    return kv_cat[..., :dim], kv_cat[..., dim:]


def _build_cache(
    pooled_by_layer: List[List[torch.Tensor]],
    full_by_layer: List[List[torch.Tensor]],
    selected_ids_by_layer: List[List[int]],
    always_full_ids: List[int],
    active_k_by_layer: List[torch.Tensor | None],
    active_v_by_layer: List[torch.Tensor | None],
    model,
) -> DynamicCache:
    cache = DynamicCache(config=model.config)

    for li in range(len(pooled_by_layer)):
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []

        selected = selected_ids_by_layer[li] if li < len(selected_ids_by_layer) else []
        selected_set = {sid for sid in selected if 0 <= sid < len(full_by_layer[li])}
        always_full_set = {sid for sid in always_full_ids if 0 <= sid < len(full_by_layer[li])}
        full_sentence_set = selected_set | always_full_set

        # Keep pooled slots only for non-selected sentences. Selected sentences are
        # represented by full token K/V below to avoid duplicated bias.
        for sid, pooled_cat in enumerate(pooled_by_layer[li]):
            if sid in full_sentence_set:
                continue
            pk, pv = _split_kv(pooled_cat)
            parts_k.append(pk)
            parts_v.append(pv)

        for sid in sorted(full_sentence_set):
            fk, fv = _split_kv(full_by_layer[li][sid])
            parts_k.append(fk)
            parts_v.append(fv)

        if active_k_by_layer[li] is not None and active_v_by_layer[li] is not None:
            parts_k.append(active_k_by_layer[li])
            parts_v.append(active_v_by_layer[li])

        if not parts_k:
            continue

        cat_k = torch.cat(parts_k, dim=2)
        cat_v = torch.cat(parts_v, dim=2)
        cache.update(cat_k, cat_v, li)

    return cache


def _append_active_token(
    final_past,
    active_k_by_layer: List[torch.Tensor | None],
    active_v_by_layer: List[torch.Tensor | None],
):
    for li, layer_past in enumerate(final_past):
        k, v = layer_past[0], layer_past[1]
        new_k = k[:, :, -1:, :]
        new_v = v[:, :, -1:, :]
        if active_k_by_layer[li] is None:
            active_k_by_layer[li] = new_k
            active_v_by_layer[li] = new_v
        else:
            active_k_by_layer[li] = torch.cat([active_k_by_layer[li], new_k], dim=2)
            active_v_by_layer[li] = torch.cat([active_v_by_layer[li], new_v], dim=2)


def _close_active_sentence(
    active_k_by_layer: List[torch.Tensor | None],
    active_v_by_layer: List[torch.Tensor | None],
    pooled_by_layer: List[List[torch.Tensor]],
    full_by_layer: List[List[torch.Tensor]],
    pool_mode: str,
):
    for li in range(len(pooled_by_layer)):
        if active_k_by_layer[li] is None or active_v_by_layer[li] is None:
            continue
        k = active_k_by_layer[li]
        v = active_v_by_layer[li]
        pk, pv = _pool_kv(k, v, pool_mode)
        pooled_by_layer[li].append(torch.cat([pk, pv], dim=-1))
        full_by_layer[li].append(torch.cat([k, v], dim=-1))
        active_k_by_layer[li] = None
        active_v_by_layer[li] = None


def _select_global_top_sentences(
    estimate_out,
    pooled_by_layer: List[List[torch.Tensor]],
    top_n: int,
) -> tuple[List[int], List[float]]:
    use_top_n = max(0, top_n)
    if use_top_n == 0 or not pooled_by_layer:
        return [], []

    sentence_counts = [len(layer_slots) for layer_slots in pooled_by_layer]
    if not sentence_counts or min(sentence_counts) == 0:
        return [], []

    # Use the common sentence index range and aggregate layer-wise cosine scores.
    num_sentences = min(sentence_counts)
    total_scores = None
    layer_hits = None

    # Preferred path: aggregate real attention mass onto pooled sentence slots.
    if estimate_out.attentions is not None:
        for attn in estimate_out.attentions:
            if attn is None:
                continue
            # attention shape: [batch, heads, q_len, kv_len]
            layer_scores = attn[0, :, -1, :num_sentences].mean(dim=0).float()
            if total_scores is None or layer_hits is None:
                total_scores = torch.zeros(num_sentences, dtype=torch.float32, device=layer_scores.device)
                layer_hits = torch.zeros(num_sentences, dtype=torch.float32, device=layer_scores.device)
            total_scores += layer_scores
            layer_hits += 1.0

    # Fallback: cosine scoring on K if attentions are unavailable.
    if total_scores is None or layer_hits is None:
        for li, layer_past in enumerate(estimate_out.past_key_values):
            query_k = layer_past[0][:, :, -1:, :].reshape(-1).float()
            query_norm = torch.norm(query_k) + 1e-8
            if total_scores is None or layer_hits is None:
                total_scores = torch.zeros(num_sentences, dtype=torch.float32, device=query_k.device)
                layer_hits = torch.zeros(num_sentences, dtype=torch.float32, device=query_k.device)

            for sid in range(num_sentences):
                pooled_cat = pooled_by_layer[li][sid]
                pooled_k, _ = _split_kv(pooled_cat)
                pooled_vec = pooled_k.reshape(-1).float()
                pooled_norm = torch.norm(pooled_vec) + 1e-8
                cosine = torch.dot(query_k, pooled_vec) / (query_norm * pooled_norm)
                total_scores[sid] += cosine
                layer_hits[sid] += 1.0

    if total_scores is None or layer_hits is None:
        return [], []

    avg_scores = total_scores / torch.clamp_min(layer_hits, 1.0)
    ranked = torch.argsort(avg_scores, descending=True).tolist()
    top_ids = ranked[: min(use_top_n, len(ranked))]
    return top_ids, avg_scores.detach().cpu().tolist()


def run_sentence_representation_inference(
    prompt: str | Sequence[dict],
    pool_mode: str,
    max_new_tokens: int,
    top_n: int,
    selection_mode: str,
    keep_last_full_prompt: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    force_close_every: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text, prompt_ids, spans, segments = _tokenize_prompt_with_spans(tokenizer, prompt)
    if not prompt_ids or not spans:
        raise ValueError("No valid sentence spans produced from prompt")

    with torch.no_grad():
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        prompt_out = model(input_ids=prompt_tensor, use_cache=True)

    pooled_by_layer, full_by_layer = _init_sentence_memory_from_prompt(prompt_out.past_key_values, spans, pool_mode=pool_mode)
    prompt_sentence_count = len(spans)
    keep_n = max(0, keep_last_full_prompt)
    always_full_ids = list(range(max(0, prompt_sentence_count - keep_n), prompt_sentence_count))

    num_layers = model.config.num_hidden_layers
    active_k_by_layer: List[torch.Tensor | None] = [None] * num_layers
    active_v_by_layer: List[torch.Tensor | None] = [None] * num_layers

    next_id = _sample_next_token(
        prompt_out.logits[0, -1, :],
        generated_ids=[],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    generated_ids: List[int] = []
    sentence_close_events = 0
    tokens_since_close = 0

    selected_snapshot: List[int] = []
    last_global_scores: List[float] = []
    fixed_selected_ids: List[int] | None = None

    for _ in range(max_new_tokens):
        cur_id = next_id
        generated_ids.append(cur_id)
        absolute_pos = len(prompt_ids) + len(generated_ids) - 1
        position_ids = torch.tensor([[absolute_pos]], device=device)
        cache_position = torch.tensor([absolute_pos], device=device)

        # Pass 1: estimate token-layer representations with pooled + active context only.
        no_retrieval = [[] for _ in range(num_layers)]
        estimate_cache = _build_cache(
            pooled_by_layer,
            full_by_layer,
            selected_ids_by_layer=no_retrieval,
            always_full_ids=always_full_ids,
            active_k_by_layer=active_k_by_layer,
            active_v_by_layer=active_v_by_layer,
            model=model,
        )

        token_tensor = torch.tensor([[cur_id]], device=device)
        if selection_mode == "once" and fixed_selected_ids is not None:
            global_selected_ids = fixed_selected_ids
            global_scores = last_global_scores
        else:
            with torch.no_grad():
                estimate_out = model(
                    input_ids=token_tensor,
                    past_key_values=estimate_cache,
                    use_cache=True,
                    output_attentions=True,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )

            global_selected_ids, global_scores = _select_global_top_sentences(
                estimate_out,
                pooled_by_layer,
                top_n=top_n,
            )
            if selection_mode == "once" and fixed_selected_ids is None:
                fixed_selected_ids = list(global_selected_ids)

        selected_ids_by_layer = [list(global_selected_ids) for _ in range(num_layers)]
        selected_snapshot = global_selected_ids
        last_global_scores = global_scores

        # Pass 2: final logits with per-layer retrieved full-token sentence K/V.
        final_cache = _build_cache(
            pooled_by_layer,
            full_by_layer,
            selected_ids_by_layer=selected_ids_by_layer,
            always_full_ids=always_full_ids,
            active_k_by_layer=active_k_by_layer,
            active_v_by_layer=active_v_by_layer,
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

        _append_active_token(final_out.past_key_values, active_k_by_layer, active_v_by_layer)
        next_id = _sample_next_token(
            final_out.logits[0, -1, :],
            generated_ids=generated_ids,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        tokens_since_close += 1

        piece = tokenizer.decode([cur_id], skip_special_tokens=False)
        force_close = force_close_every > 0 and tokens_since_close >= force_close_every
        if any(x in piece for x in [".", "!", "?"]) or force_close:
            _close_active_sentence(
                active_k_by_layer,
                active_v_by_layer,
                pooled_by_layer,
                full_by_layer,
                pool_mode=pool_mode,
            )
            sentence_close_events += 1
            tokens_since_close = 0

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = prompt_text + generated_text

    completed_sentence_count = len(pooled_by_layer[0]) if pooled_by_layer else 0
    active_tokens_now = 0
    if active_k_by_layer and active_k_by_layer[0] is not None:
        active_tokens_now = int(active_k_by_layer[0].shape[2])

    # Pooled sentence slot + retrieved full slots + active sentence tokens per layer.
    avg_selected = float(len(selected_snapshot))
    estimated_kv_slots = int(num_layers * (completed_sentence_count + active_tokens_now + avg_selected))

    return {
        "pool_mode": pool_mode,
        "top_n": top_n,
        "segments": segments,
        "completed_sentence_slots": completed_sentence_count,
        "active_sentence_tokens": active_tokens_now,
        "sentence_close_events": sentence_close_events,
        "generated_tokens": len(generated_ids),
        "generated_text": generated_text,
        "full_text": full_text,
        "estimated_kv_slots": estimated_kv_slots,
        "last_step_selected_sentence_ids": selected_snapshot,
        "last_step_global_sentence_scores": last_global_scores,
        "decode_temperature": temperature,
        "decode_top_p": top_p,
        "decode_repetition_penalty": repetition_penalty,
        "force_close_every": force_close_every,
        "selection_mode": selection_mode,
        "keep_last_full_prompt": keep_last_full_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Sentence pooled + per-layer sentence retrieval inference")
    parser.add_argument("--pool", choices=["mean", "max"], default="mean", help="Pooling mode for per-sentence K/V")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="Top N sentence pools to expand to full K/V per layer")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--selection-mode", choices=["once", "token"], default="once")
    parser.add_argument("--keep-last-full-prompt", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--force-close-every", type=int, default=24)
    args = parser.parse_args()

    prompt = [
        {"content": "Hi there", "role": "user"},
        {"content": "Hello! How can I help you today?", "role": "assistant"},
        {
            "content": "I'm looking for a beach resort for my next vacation. Can you recommend some popular ones?",
            "role": "user",
        },
    ]

    result = run_sentence_representation_inference(
        prompt,
        pool_mode=args.pool,
        max_new_tokens=args.max_new_tokens,
        top_n=args.top_n,
        selection_mode=args.selection_mode,
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
    print("Last-step selected sentence ids (shared across layers):", result["last_step_selected_sentence_ids"])
    print("Last-step global sentence scores:", result["last_step_global_sentence_scores"])
    print("Selection mode:", result["selection_mode"])
    print("Keep last full prompt sentences:", result["keep_last_full_prompt"])
    print("Decode temperature:", result["decode_temperature"])
    print("Decode top-p:", result["decode_top_p"])
    print("Decode repetition penalty:", result["decode_repetition_penalty"])
    print("Force-close-every tokens:", result["force_close_every"])
    print("Generated text:", result["generated_text"])
    print("Full text:", result["full_text"])


if __name__ == "__main__":
    main()
