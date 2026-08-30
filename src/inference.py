import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import SentenceSparseSmolLM2ForCausalLM
from sentence import SENT_TOKEN, add_sentence_tokens, add_sentence_tokens_from_messages


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
FINETUNED_DIR = "./sentence-sparse-smollm2-135m"
MAX_NEW_TOKENS = 30


# Count full attention scores
def count_full_attention_scores(seq_len, num_layers):
    return num_layers * (seq_len * (seq_len + 1) // 2)


# Count full kv slots
def count_full_kv_slots(seq_len, num_layers):
    return num_layers * seq_len


# Count cache-style full decode attention scores
def count_full_decode_attention_scores(prompt_len, new_tokens, num_layers):
    total = 0
    cur_len = prompt_len
    for _ in range(new_tokens):
        total += num_layers * cur_len
        cur_len += 1
    return total


# Find latest epoch checkpoint
def latest_checkpoint(path):
    cks = sorted(glob.glob(os.path.join(path, "checkpoint-epoch-*")))
    return cks[-1] if cks else path


# Baseline full attention generation
def run_full_attention(prompt, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if isinstance(prompt, list):
        inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    prompt_len = int(inputs["input_ids"].shape[1])
    seq_len = out.shape[1]
    generated_tokens = int(seq_len - prompt_len)
    full_attn_scores_final_seq = count_full_attention_scores(seq_len, model.config.num_hidden_layers)
    full_attn_scores_decode_cache = count_full_decode_attention_scores(prompt_len, generated_tokens, model.config.num_hidden_layers)
    full_kv_slots = count_full_kv_slots(seq_len, model.config.num_hidden_layers)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"text": text, "seq_len": seq_len, "prompt_len": prompt_len, "generated_tokens": generated_tokens, "attn_scores_final_seq": full_attn_scores_final_seq, "attn_scores_decode_cache": full_attn_scores_decode_cache, "kv_slots": full_kv_slots}


# Build sparse initial token state
def initial_sparse_state(text, tokenizer):
    marked = text
    ids = tokenizer(marked, return_tensors="pt").input_ids[0].tolist()
    sent_id = tokenizer.convert_tokens_to_ids(SENT_TOKEN)
    sent_pos = [i for i, t in enumerate(ids) if t == sent_id]
    if not sent_pos:
        return [], [sent_id] + ids
    last = sent_pos[-1]
    completed = []
    for i in sent_pos[:-1]:
        completed.append(sent_id)
    active = ids[last:]
    return completed, active


# Sparse sentence generation simulation
def run_sparse_attention(prompt, device):
    ckpt = latest_checkpoint(FINETUNED_DIR)
    if os.path.exists(ckpt):
        print(f"Using checkpoint: {ckpt}")
    else:
        print(f"No checkpoint found in {FINETUNED_DIR}, using base model: {MODEL_ID}!!!!!!")
    tokenizer = AutoTokenizer.from_pretrained(ckpt if os.path.exists(ckpt) else MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if SENT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [SENT_TOKEN]})

    model = SentenceSparseSmolLM2ForCausalLM.from_pretrained(ckpt if os.path.exists(ckpt) else MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    sent_id = tokenizer.convert_tokens_to_ids(SENT_TOKEN)
    with torch.no_grad():
        ref_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        model.model.embed_tokens.weight[sent_id].copy_(model.model.embed_tokens.weight[ref_id])
        model.lm_head.weight[sent_id].copy_(model.lm_head.weight[ref_id])
    model.set_sentence_token_id(sent_id)
    model.to(device)
    model.eval()

    if isinstance(prompt, list):
        prompt_text = add_sentence_tokens_from_messages(prompt)
    else:
        prompt_text = add_sentence_tokens(prompt)
    completed, active = initial_sparse_state(prompt_text, tokenizer)
    generated = []
    sparse_attn_scores_decode_cache = 0
    sparse_kv_slots_running = 0
    num_layers = model.config.num_hidden_layers
    dropped_word_tokens_total = 0
    sentence_flush_events = 0
    peak_active_word_tokens = max(len(active) - 1, 0)

    for _ in range(MAX_NEW_TOKENS):
        ids = completed + active
        input_ids = torch.tensor([ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Approximate cache-style decode attention for one new token query.
        # Non-last layers: current token -> all sentence tokens + active sentence tokens.
        # Last layer: current token -> active sentence tokens only.
        num_sentence_tokens = len(completed) + 1
        active_len = len(active)
        keys_non_last = num_sentence_tokens + max(active_len - 1, 0)
        keys_last = active_len
        sparse_attn_scores_decode_cache += (num_layers - 1) * keys_non_last + keys_last
        sparse_kv_slots_running += num_layers * len(ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        next_id = int(torch.argmax(out.logits[0, -1, :]).item())
        generated.append(next_id)
        active.append(next_id)
        peak_active_word_tokens = max(peak_active_word_tokens, max(len(active) - 1, 0))

        piece = tokenizer.decode([next_id], skip_special_tokens=False)
        if any(x in piece for x in [".", "!", "?"]):
            dropped_word_tokens_total += max(len(active) - 1, 0)
            sentence_flush_events += 1
            completed.append(sent_id)
            active = [sent_id]

    final_ids = completed + active
    text = tokenizer.decode(final_ids, skip_special_tokens=True)
    kv_sentence_only = num_layers * (len(completed) + len(active))
    active_word_tokens_now = max(len(active) - 1, 0)
    return {"text": text, "context_tokens": len(completed) + len(active), "attn_scores_decode_cache": sparse_attn_scores_decode_cache, "kv_slots_running": sparse_kv_slots_running, "kv_slots_sentence_only": kv_sentence_only, "active_word_tokens_now": active_word_tokens_now, "peak_active_word_tokens": peak_active_word_tokens, "dropped_word_tokens_total": dropped_word_tokens_total, "sentence_flush_events": sentence_flush_events}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = [
        {"content": "Hi there", 
         "role": "user"
        }, 
        {"content": "Hello! How can I help you today?", 
         "role": "assistant"
        }, 
        {"content": "I'm looking for a beach resort for my next vacation. Can you recommend some popular ones?", 
         "role": "user"
        }
    ]

    full = run_full_attention(prompt, device)
    sparse = run_sparse_attention(prompt, device)

    print("==== Full Attention SmolLM2 ====")
    print("Output:", full["text"])
    print("Prompt tokens:", full["prompt_len"])
    print("Generated tokens:", full["generated_tokens"])
    print("Final tokens:", full["seq_len"])
    print("Attention scores (final full matrix):", full["attn_scores_final_seq"])
    print("Attention scores (decode cache-style):", full["attn_scores_decode_cache"])
    print("KV slots created:", full["kv_slots"])

    print("\n==== Sentence Sparse Fine-tuned Model ====")
    print("Output:", sparse["text"])
    print("Current context tokens:", sparse["context_tokens"])
    print("Attention scores (decode cache-style):", sparse["attn_scores_decode_cache"])
    print("KV slots created during run:", sparse["kv_slots_running"])
    print("KV slots with sentence-only memory:", sparse["kv_slots_sentence_only"])
    print("Active sentence word-token cache now:", sparse["active_word_tokens_now"])
    print("Peak active sentence word-token cache:", sparse["peak_active_word_tokens"])
    print("Dropped word tokens after sentence end:", sparse["dropped_word_tokens_total"])
    print("Sentence completion flush events:", sparse["sentence_flush_events"])


if __name__ == "__main__":
    main()
