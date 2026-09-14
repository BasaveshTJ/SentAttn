import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from sentence import split_chat_sentences, split_sentences


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_NEW_TOKENS = 100


def pool_kv(k, v, mode):
    if mode == "mean":
        return k.mean(dim=2, keepdim=True), v.mean(dim=2, keepdim=True)
    if mode == "max":
        return k.max(dim=2, keepdim=True).values, v.max(dim=2, keepdim=True).values
    raise ValueError(f"Unknown pooling mode: {mode}")


def find_subsequence(haystack, needle, start):
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    for i in range(start, max(start, limit)):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def get_prompt_ids_and_spans(tokenizer, prompt):
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0].tolist()
        chunks = [seg.split(": ", 1)[1] if ": " in seg else seg for seg in split_chat_sentences(prompt)]
    else:
        text = prompt
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        chunks = split_sentences(prompt)

    spans = []
    cursor = 0
    for chunk in chunks:
        found = -1
        found_len = 0
        for variant in (chunk, " " + chunk, "\n" + chunk):
            token_ids = tokenizer(variant, add_special_tokens=False).input_ids
            pos = find_subsequence(ids, token_ids, cursor)
            if pos != -1:
                found = pos
                found_len = len(token_ids)
                break
        if found == -1:
            continue
        spans.append((found, found + found_len))
        cursor = found + found_len
    return text, ids, spans


def compress_prompt_cache(past_key_values, spans, pool_mode, model, keep_last_n_uncompressed):
    cache = DynamicCache(config=model.config)
    keep_n = max(0, keep_last_n_uncompressed)
    pool_until = max(0, len(spans) - keep_n)

    for layer_idx, layer_past in enumerate(past_key_values):
        k, v = layer_past[0], layer_past[1]
        parts_k, parts_v = [], []
        cursor = 0

        for sent_idx, (start, end) in enumerate(spans):
            if start > cursor:
                parts_k.append(k[:, :, cursor:start, :])
                parts_v.append(v[:, :, cursor:start, :])

            seg_k = k[:, :, start:end, :]
            seg_v = v[:, :, start:end, :]
            if sent_idx < pool_until:
                seg_k, seg_v = pool_kv(seg_k, seg_v, pool_mode)
            parts_k.append(seg_k)
            parts_v.append(seg_v)
            cursor = end

        if cursor < int(k.shape[2]):
            parts_k.append(k[:, :, cursor:, :])
            parts_v.append(v[:, :, cursor:, :])

        cache.update(torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2), layer_idx)
    return cache


def run_sentence_pool_inference(prompt, pool_mode, max_new_tokens, keep_last_n_uncompressed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    prompt_text, prompt_ids, spans = get_prompt_ids_and_spans(tokenizer, prompt)
    if not prompt_ids or not spans:
        raise ValueError("No valid sentence spans found in prompt")

    with torch.no_grad():
        prompt_out = model(input_ids=torch.tensor([prompt_ids], device=device), use_cache=True)

    cache = compress_prompt_cache(
        prompt_out.past_key_values,
        spans,
        pool_mode=pool_mode,
        model=model,
        keep_last_n_uncompressed=keep_last_n_uncompressed,
    )

    generated = []
    next_id = int(prompt_out.logits[0, -1, :].argmax())
    for _ in range(max_new_tokens):
        cur = torch.tensor([[next_id]], device=device)
        position = torch.tensor([[cache.get_seq_length(0)]], device=device)
        with torch.no_grad():
            out = model(input_ids=cur, past_key_values=cache, use_cache=True, position_ids=position)
        cache = out.past_key_values
        generated.append(next_id)
        next_id = int(out.logits[0, -1, :].argmax())

    return {
        "pool_mode": pool_mode,
        "prompt_sentence_slots": len(spans),
        "keep_last_n_uncompressed": keep_last_n_uncompressed,
        "generated_tokens": len(generated),
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "full_text": prompt_text + tokenizer.decode(generated, skip_special_tokens=True),
        "estimated_kv_slots": model.config.num_hidden_layers * cache.get_seq_length(0),
    }


def main():
    parser = argparse.ArgumentParser(description="Sentence pooled KV-cache inference prototype")
    parser.add_argument("--pool", choices=["mean", "max"], default="mean")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--keep-last-n-uncompressed", type=int, default=0)
    args = parser.parse_args()

    prompt = [{"content": "Hi there", "role": "user"}]
    result = run_sentence_pool_inference(
        prompt,
        pool_mode=args.pool,
        max_new_tokens=args.max_new_tokens,
        keep_last_n_uncompressed=args.keep_last_n_uncompressed,
    )

    print("=== Sentence-Pooled KV Inference ===")
    print("Pool mode:", result["pool_mode"])
    print("Prompt sentence slots:", result["prompt_sentence_slots"])
    print("Keep last N uncompressed:", result["keep_last_n_uncompressed"])
    print("Generated tokens:", result["generated_tokens"])
    print("Estimated KV slots kept:", result["estimated_kv_slots"])
    print("Generated text:", result["generated_text"])
    print("Full text:", result["full_text"])


if __name__ == "__main__":
    main()
