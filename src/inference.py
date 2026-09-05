import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import SentenceSparseSmolLM2ForCausalLM
from sentence import SENT_TOKEN, add_sentence_tokens, add_sentence_tokens_from_messages


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
# FINETUNED_DIR = "./sentence-sparse-smollm2-135m"
# FINETUNED_DIR = "./sentence-sparse-smollm2-135m_edbd445"
# FINETUNED_DIR = "./sentence-sparse-smollm2-135m_66d102a"
# FINETUNED_DIR = "./sentence-sparse-smollm2-135m_7fcbae4"
# FINETUNED_DIR = "./sentence-sparse-smollm2-135m_7fcbae4_edc"
FINETUNED_DIR = "./sentence-local-global-smollm2-135m/checkpoint-432"
MAX_NEW_TOKENS = 100


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
    generated_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return {
        "model": "full_attention",
        "text": text,
        "generated_text": generated_text,
        "prompt_tokens": prompt_len,
        "generated_tokens": generated_tokens,
        "final_tokens": seq_len,
        "attn_scores_final_seq": full_attn_scores_final_seq,
        "attn_scores_decode_cache": full_attn_scores_decode_cache,
        "kv_slots_created": full_kv_slots,
    }


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
    model.set_sentence_token_id(sent_id)
    model.to(device)
    model.eval()

    if isinstance(prompt, list):
        prompt_text = add_sentence_tokens_from_messages(prompt)
        base_prompt_text = " ".join([f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in prompt])
    else:
        base_prompt_text = prompt
        prompt_text = add_sentence_tokens(base_prompt_text)

    # Local attention only ever looks within the current sentence segment, so once a
    # sentence closes (model predicts a new <|sent|>), its word tokens can never be
    # attended to again -- only its <|sent|> id needs to stick around for the global
    # (sent-to-sent) pass. So we drop those word tokens instead of keeping them.
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
    prompt_tokens = len(ids)
    sent_positions = [i for i, t in enumerate(ids) if t == sent_id]
    if sent_positions:
        completed = [ids[i] for i in sent_positions[:-1]]
        active = ids[sent_positions[-1]:]
    else:
        completed, active = [], ids

    generated = []
    num_layers = model.config.num_hidden_layers

    for _ in range(MAX_NEW_TOKENS):
        cur_ids = completed + active
        input_ids = torch.tensor([cur_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        next_id = int(torch.argmax(out.logits[0, -1, :]).item())
        generated.append(next_id)

        if next_id == sent_id:
            # Sentence just closed: keep only its <|sent|> id, drop its word tokens.
            completed.append(active[0])
            active = [next_id]
        else:
            active.append(next_id)

    final_ids = completed + active
    generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not generated_text:
        generated_text = tokenizer.decode(generated, skip_special_tokens=False).replace(SENT_TOKEN, "").strip()
    text = (base_prompt_text + generated_text).strip()
    seq_len = len(final_ids)
    return {
        "model": "sentence_sparse",
        "text": text,
        "generated_text": generated_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated),
        "final_tokens": seq_len,
        "attn_scores_final_seq": None,
        "attn_scores_decode_cache": None,
        "kv_slots_created": num_layers * seq_len,
    }


def print_metrics(name, metrics):
    print(f"==== {name} ====")
    print("Generated text:", metrics["generated_text"])
    print("Prompt tokens:", metrics["prompt_tokens"])
    print("Generated tokens:", metrics["generated_tokens"])
    print("Final tokens:", metrics["final_tokens"])
    print("Attention scores (final full matrix):", metrics["attn_scores_final_seq"])
    print("Attention scores (decode cache-style):", metrics["attn_scores_decode_cache"])
    print("KV slots created:", metrics["kv_slots_created"])


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

    print_metrics("Full Attention SmolLM2", full)
    print()
    print_metrics("Sentence Sparse Fine-tuned Model", sparse)


if __name__ == "__main__":
    main()
