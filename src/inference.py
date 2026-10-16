import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_withKVcache import SentenceSparseSmolLM2WithKVCache
from sentence import (
    SENT_TOKEN,
    add_sentence_tokens,
    add_system_sentence_token,
    add_sentence_tokens_to_messages,
)


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
FINETUNED_DIR = "./sentence-sparse-smollm2-135m-edc"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.4
TOP_P = 0.95


def cache_kv_tokens(past_key_values):
    """Number of tokens currently stored in a cache (pruned or dense)."""
    if past_key_values is None:
        return 0
    token_ids = getattr(past_key_values, "_token_ids", None)
    if token_ids is not None:
        return token_ids.shape[1]
    layers = getattr(past_key_values, "layers", None)
    if layers:
        return layers[0].keys.shape[-2]
    return 0


def causal_attention_count(cache_before, query_length):
    """Number of (query, key) pairs a dense causal mask attends to."""
    return cache_before * query_length + query_length * (query_length + 1) // 2


def track_generation_stats(model):
    """Record cache size and attention-score usage for every forward call.

    Works for both plain dense-causal generation and the sparse model: the
    sparse model's prefill call (the only step with a non-trivial mask) uses
    its own `allowed_mask()` to get the exact attended-pair count; every other
    step (decode, or full attention) is dense causal, so a closed-form count
    is used instead of materializing the mask.
    """
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_attention_heads)
    stats = {"cache_sizes": [], "attention_scores": []}

    def hook(module, _args, kwargs, output):
        input_ids = kwargs.get("input_ids")
        query_length = input_ids.shape[1]
        cache_before = cache_kv_tokens(kwargs.get("past_key_values"))

        if query_length > 1 and cache_before == 0 and hasattr(module, "allowed_mask"):
            pair_count = int(module.allowed_mask(input_ids).sum().item())
        else:
            pair_count = causal_attention_count(cache_before, query_length)

        stats["cache_sizes"].append(cache_kv_tokens(getattr(output, "past_key_values", None)))
        stats["attention_scores"].append(layers * heads * pair_count)

    handle = model.register_forward_hook(hook, with_kwargs=True)
    return stats, handle


def resolve_checkpoint_dirs(path):
    # If path is a specific checkpoint, its tokenizer lives in the parent dir.
    # If path is the parent dir, resolve to its latest checkpoint.
    if os.path.basename(os.path.normpath(path)).startswith("checkpoint-"):
        return path, os.path.dirname(os.path.normpath(path))
    checkpoints = sorted(glob.glob(os.path.join(path, "checkpoint-*")))
    return (checkpoints[-1] if checkpoints else path), path


def load_sparse_model(device):
    checkpoint_dir, tokenizer_dir = resolve_checkpoint_dirs(FINETUNED_DIR)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Sparse model path not found: {checkpoint_dir}. Train or point FINETUNED_DIR to a valid sparse checkpoint."
        )
    if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        raise FileNotFoundError(
            f"Sparse tokenizer files not found in {tokenizer_dir}. Save tokenizer with sparse model and retry."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": [SENT_TOKEN]})
    model = SentenceSparseSmolLM2WithKVCache.from_pretrained(
        checkpoint_dir, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.set_sentence_token_id(tokenizer.convert_tokens_to_ids(SENT_TOKEN))
    structural = [tokenizer.convert_tokens_to_ids(t) for t in (
        "<|im_start|>", "<|im_end|>", "system", "user", "assistant"
    )]
    model.set_structural_token_ids(structural)
    return model.to(device).eval(), tokenizer


def load_full_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    return model, tokenizer


def run_full_attention(prompt, device):
    model, tokenizer = load_full_model(device)
    if isinstance(prompt, list):
        encoded = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = (encoded.input_ids if hasattr(encoded, "input_ids") else encoded).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    stats, handle = track_generation_stats(model)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    handle.remove()

    generated = outputs[0, input_ids.shape[1]:].tolist()
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=False).strip(),
        "generated_tokens": len(generated),
        "max_cache_kv_tokens": max(stats["cache_sizes"], default=0),
        "last_cache_kv_tokens": stats["cache_sizes"][-1] if stats["cache_sizes"] else 0,
        "cumulative_attention_scores": sum(stats["attention_scores"]),
        "Last_Token_Attention_Scores": stats["attention_scores"][-1] if stats["attention_scores"] else 0,
    }


def run_sparse_attention(prompt, device):
    model, tokenizer = load_sparse_model(device)
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            add_sentence_tokens_to_messages(prompt),
            tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(add_system_sentence_token(text)).input_ids
    else:
        ids = tokenizer(add_sentence_tokens(prompt)).input_ids

    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    eos_token_id = tokenizer.eos_token_id

    stats, handle = track_generation_stats(model)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        use_cache=True,
    )
    handle.remove()

    generated = outputs[0, input_ids.shape[1]:].tolist()
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=False).strip(),
        "generated_tokens": len(generated),
        "max_cache_kv_tokens": max(stats["cache_sizes"], default=0),
        "last_cache_kv_tokens": stats["cache_sizes"][-1] if stats["cache_sizes"] else 0,
        "cumulative_attention_scores": sum(stats["attention_scores"]),
        "Last_Token_Attention_Scores": stats["attention_scores"][-1] if stats["attention_scores"] else 0,
    }


if __name__ == "__main__":
    prompt = [
        {
            "content": "which is better, Jamaica or the Bahamas?",
            "role": "user"
        }
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = run_full_attention(prompt, device)
    sparse = run_sparse_attention(prompt, device)

    print("HuggingFaceTB/SmolLM2-135M-Instruct model results:")
    print("Full text:", full["text"])
    print("Full generated tokens:", full["generated_tokens"])
    print("Full max cache kv tokens:", full["max_cache_kv_tokens"])
    print("Full last cache kv tokens:", full["last_cache_kv_tokens"])
    print("Full cumulative attention scores:", full["cumulative_attention_scores"])
    print("Full Last Token Attention Scores:", full["Last_Token_Attention_Scores"])

    print("\n\nSparse attention results:")
    print("Sparse text:", sparse["text"])
    print("Sparse generated tokens:", sparse["generated_tokens"])
    print("Sparse max cache kv tokens:", sparse["max_cache_kv_tokens"])
    print("Sparse last cache kv tokens:", sparse["last_cache_kv_tokens"])
    print("Sparse cumulative attention scores:", sparse["cumulative_attention_scores"])
    print("Sparse Last Token Attention Scores:", sparse["Last_Token_Attention_Scores"])