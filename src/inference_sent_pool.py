import argparse
from typing import List, Sequence

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

from sentence import split_chat_sentences, split_sentences


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_NEW_TOKENS = 100


def _pool_kv(k: torch.Tensor, v: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    # k/v shape: [batch, heads, seq, head_dim]
    if mode == "mean":
        return k.mean(dim=2, keepdim=True), v.mean(dim=2, keepdim=True)
    if mode == "max":
        return k.max(dim=2, keepdim=True).values, v.max(dim=2, keepdim=True).values
    raise ValueError(f"Unknown pooling mode: {mode}")


def _sentence_segments_from_prompt(prompt: str | Sequence[dict]) -> List[str]:
    if isinstance(prompt, list):
        return split_chat_sentences(prompt)
    return split_sentences(prompt)


def _find_subsequence(haystack: List[int], needle: List[int], start_idx: int) -> int:
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    for i in range(start_idx, max(limit, start_idx)):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _tokenize_prompt_with_spans(tokenizer, prompt: str | Sequence[dict]) -> tuple[str, List[int], List[tuple[int, int]]]:
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
        sentence_chunks: List[str] = []
        for msg in prompt:
            content = msg.get("content", "")
            parts = split_sentences(content)
            if not parts:
                clean = content.strip()
                if clean:
                    sentence_chunks.append(clean)
            else:
                sentence_chunks.extend(parts)

        spans: List[tuple[int, int]] = []
        cursor = 0
        for chunk in sentence_chunks:
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
        return prompt_text, prompt_ids, spans

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
    return prompt_text, prompt_ids, spans


def _compress_cache_by_spans(past_key_values, spans: Sequence[tuple[int, int]], pool_mode: str, model, keep_last_n_uncompressed: int) -> DynamicCache:
    pooled_cache = DynamicCache(config=model.config)
    if not spans:
        return pooled_cache

    total = len(spans)
    pooled_count = max(0, total - max(0, keep_last_n_uncompressed))

    for li, layer_past in enumerate(past_key_values):
        k, v = layer_past[0], layer_past[1]
        pieces_k: List[torch.Tensor] = []
        pieces_v: List[torch.Tensor] = []
        cursor = 0
        for idx, (start, end) in enumerate(spans):
            if start > cursor:
                pieces_k.append(k[:, :, cursor:start, :])
                pieces_v.append(v[:, :, cursor:start, :])

            seg_k = k[:, :, start:end, :]
            seg_v = v[:, :, start:end, :]
            if idx < pooled_count:
                pk, pv = _pool_kv(seg_k, seg_v, pool_mode)
                pieces_k.append(pk)
                pieces_v.append(pv)
            else:
                pieces_k.append(seg_k)
                pieces_v.append(seg_v)
            cursor = end

        if cursor < int(k.shape[2]):
            pieces_k.append(k[:, :, cursor:, :])
            pieces_v.append(v[:, :, cursor:, :])

        new_k = torch.cat(pieces_k, dim=2)
        new_v = torch.cat(pieces_v, dim=2)
        pooled_cache.update(new_k, new_v, li)

    return pooled_cache


def _compress_oldest_completed_generated(cache: DynamicCache, oldest_len: int, rest_uncompressed_len: int, pool_mode: str, model) -> DynamicCache:
    if oldest_len <= 0:
        return cache

    compressed = DynamicCache(config=model.config)
    for li, layer_past in enumerate(cache):
        k, v = layer_past[0], layer_past[1]
        seq_len = int(k.shape[2])
        tail_total = oldest_len + rest_uncompressed_len
        if tail_total > seq_len:
            raise ValueError("completed sentence tail exceeds cache sequence length")

        start = seq_len - tail_total
        end = start + oldest_len
        prefix_k = k[:, :, :start, :] if start > 0 else None
        prefix_v = v[:, :, :start, :] if start > 0 else None
        oldest_k = k[:, :, start:end, :]
        oldest_v = v[:, :, start:end, :]
        suffix_k = k[:, :, end:, :] if end < seq_len else None
        suffix_v = v[:, :, end:, :] if end < seq_len else None
        pooled_k, pooled_v = _pool_kv(oldest_k, oldest_v, pool_mode)

        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []
        if prefix_k is not None:
            parts_k.append(prefix_k)
            parts_v.append(prefix_v)
        parts_k.append(pooled_k)
        parts_v.append(pooled_v)
        if suffix_k is not None:
            parts_k.append(suffix_k)
            parts_v.append(suffix_v)

        new_k = torch.cat(parts_k, dim=2)
        new_v = torch.cat(parts_v, dim=2)
        compressed.update(new_k, new_v, li)

    return compressed


def _compress_active_tail(cache: DynamicCache, active_len: int, pool_mode: str, model) -> DynamicCache:
    if active_len <= 0:
        return cache

    compressed = DynamicCache(config=model.config)
    for li, layer_past in enumerate(cache):
        k, v = layer_past[0], layer_past[1]
        seq_len = int(k.shape[2])
        if active_len > seq_len:
            raise ValueError("active sentence length exceeds cache sequence length")

        prefix_len = seq_len - active_len
        prefix_k = k[:, :, :prefix_len, :] if prefix_len > 0 else None
        prefix_v = v[:, :, :prefix_len, :] if prefix_len > 0 else None
        active_k = k[:, :, prefix_len:, :]
        active_v = v[:, :, prefix_len:, :]
        pooled_k, pooled_v = _pool_kv(active_k, active_v, pool_mode)

        if prefix_k is not None:
            new_k = torch.cat([prefix_k, pooled_k], dim=2)
            new_v = torch.cat([prefix_v, pooled_v], dim=2)
        else:
            new_k = pooled_k
            new_v = pooled_v

        compressed.update(new_k, new_v, li)

    return compressed


def _past_len(past_key_values) -> int:
    if not past_key_values:
        return 0
    return int(past_key_values.get_seq_length(0))


def run_sentence_pool_inference(prompt: str | Sequence[dict], pool_mode: str, max_new_tokens: int, keep_last_n_uncompressed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    segments = _sentence_segments_from_prompt(prompt)
    prompt_text, prompt_ids, spans = _tokenize_prompt_with_spans(tokenizer, prompt)
    if not prompt_ids or not spans:
        raise ValueError("No valid sentence segments produced from prompt")

    with torch.no_grad():
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        prompt_out = model(input_ids=prompt_tensor, use_cache=True)

    # Start from the real prompt cache, then compress each sentence into one pooled KV slot.
    cache = _compress_cache_by_spans(
        prompt_out.past_key_values,
        spans,
        pool_mode=pool_mode,
        model=model,
        keep_last_n_uncompressed=keep_last_n_uncompressed,
    )

    next_id = int(torch.argmax(prompt_out.logits[0, -1, :]).item())
    generated_ids: List[int] = []
    active_sentence_len = 0
    prompt_sentence_slots = len(spans)
    completed_generated_sentences = 0
    uncompressed_completed_generated_lens: List[int] = []

    for _ in range(max_new_tokens):
        cur_id = next_id
        generated_ids.append(cur_id)
        active_sentence_len += 1

        token_tensor = torch.tensor([[cur_id]], device=device)
        pos_start = _past_len(cache)
        position_ids = torch.arange(
            pos_start,
            pos_start + token_tensor.shape[1],
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            out = model(
                input_ids=token_tensor,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )

        cache = out.past_key_values
        next_id = int(torch.argmax(out.logits[0, -1, :]).item())

        piece = tokenizer.decode([cur_id], skip_special_tokens=False)
        if any(x in piece for x in [".", "!", "?"]):
            completed_generated_sentences += 1
            uncompressed_completed_generated_lens.append(active_sentence_len)
            active_sentence_len = 0

            # Keep recent completed generated sentences uncompressed; compress only older ones.
            while len(uncompressed_completed_generated_lens) > max(0, keep_last_n_uncompressed):
                oldest_len = uncompressed_completed_generated_lens.pop(0)
                rest_len = sum(uncompressed_completed_generated_lens)
                cache = _compress_oldest_completed_generated(
                    cache,
                    oldest_len=oldest_len,
                    rest_uncompressed_len=rest_len,
                    pool_mode=pool_mode,
                    model=model,
                )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = prompt_text + generated_text

    num_layers = model.config.num_hidden_layers
    final_sentence_slots = prompt_sentence_slots + completed_generated_sentences
    final_active = active_sentence_len
    total_kv_slots_per_layer = final_sentence_slots + final_active
    estimated_kv_slots = num_layers * total_kv_slots_per_layer

    return {
        "pool_mode": pool_mode,
        "segments": segments,
        "prompt_sentence_slots": prompt_sentence_slots,
        "completed_generated_sentences": completed_generated_sentences,
        "keep_last_n_uncompressed": keep_last_n_uncompressed,
        "active_sentence_tokens": final_active,
        "generated_tokens": len(generated_ids),
        "generated_text": generated_text,
        "full_text": full_text,
        "estimated_kv_slots": estimated_kv_slots,
    }


def main():
    parser = argparse.ArgumentParser(description="Sentence pooled KV-cache inference prototype")
    parser.add_argument("--pool", choices=["mean", "max"], default="mean", help="Pooling mode for per-sentence K/V")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--keep-last-n-uncompressed", type=int, default=0, help="Keep this many most recent completed sentences uncompressed")
    args = parser.parse_args()

    prompt = [
        {"content": "Hi there", "role": "user"},
        {"content": "Hello! How can I help you today?", "role": "assistant"},
        {
            "content": "I'm looking for a beach resort for my next vacation. Can you recommend some popular ones?",
            "role": "user",
        },
    ]

    result = run_sentence_pool_inference(
        prompt,
        pool_mode=args.pool,
        max_new_tokens=args.max_new_tokens,
        keep_last_n_uncompressed=args.keep_last_n_uncompressed,
    )

    print("=== Sentence-Pooled KV Inference ===")
    print("Pool mode:", result["pool_mode"])
    print("Keep last N uncompressed:", result["keep_last_n_uncompressed"])
    print("Sentence segments:", result["segments"])
    print("Prompt sentence slots:", result["prompt_sentence_slots"])
    print("Completed generated sentence slots:", result["completed_generated_sentences"])
    print("Active sentence token count:", result["active_sentence_tokens"])
    print("Generated tokens:", result["generated_tokens"])
    print("Estimated KV slots kept:", result["estimated_kv_slots"])
    print("Generated text:", result["generated_text"])
    print("Full text:", result["full_text"])


if __name__ == "__main__":
    main()
