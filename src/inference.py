import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import SentenceSparseSmolLM2ForCausalLM
from sentence import (
    SENT_TOKEN,
    add_sentence_tokens,
    add_system_sentence_token,
    add_sentence_tokens_to_messages,
)


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
FINETUNED_DIR = "./sentence-sparse-smollm2-135m-all"
MAX_NEW_TOKENS = 100

def attention_scores_per_step(seq_len, model):
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_attention_heads)
    return layers * heads * seq_len * seq_len


def latest_checkpoint(path):
    checkpoints = sorted(glob.glob(os.path.join(path, "checkpoint-*")))
    return checkpoints[-1] if checkpoints else path


def load_sparse_model(device):
    path = latest_checkpoint(FINETUNED_DIR)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sparse model path not found: {path}. Train or point FINETUNED_DIR to a valid sparse checkpoint."
        )
    source = path
    tokenizer_source = source if os.path.exists(os.path.join(source, "tokenizer.json")) else FINETUNED_DIR
    if not os.path.exists(os.path.join(tokenizer_source, "tokenizer.json")):
        raise FileNotFoundError(
            "Sparse tokenizer files not found in checkpoint or FINETUNED_DIR. "
            "Save tokenizer with sparse model and retry."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.add_special_tokens({"additional_special_tokens": [SENT_TOKEN]})
    model = SentenceSparseSmolLM2ForCausalLM.from_pretrained(
        source, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.set_sentence_token_id(tokenizer.convert_tokens_to_ids(SENT_TOKEN))
    structural = [tokenizer.convert_tokens_to_ids(t) for t in (
        "<|im_start|>", "<|im_end|>", "system", "user", "assistant"
    )]
    model.set_structural_token_ids(structural)
    return model.to(device).eval(), tokenizer


def run_full_attention(prompt, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    if isinstance(prompt, list):
        inputs = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    current = inputs["input_ids"]
    generated = []
    cumulative_attention_scores = 0

    for _ in range(MAX_NEW_TOKENS):
        seq_len = current.shape[1]
        cumulative_attention_scores += attention_scores_per_step(seq_len, model)
        with torch.no_grad():
            logits = model(input_ids=current, attention_mask=torch.ones_like(current)).logits
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=device, dtype=current.dtype)
        current = torch.cat([current, next_token], dim=1)

    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "cumulative_attention_scores": int(cumulative_attention_scores),
        "generated_tokens": len(generated),
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
    sent_id = tokenizer.convert_tokens_to_ids(SENT_TOKEN)
    structural = {
        tokenizer.convert_tokens_to_ids(t)
        for t in ("<|im_start|>", "<|im_end|>", "system", "user", "assistant")
    }
    last_sent = max((i for i, token in enumerate(ids) if token == sent_id), default=-1)
    completed = [token for token in ids[:last_sent + 1] if token in structural or token == sent_id]
    active = ids[last_sent + 1:]
    generated = []
    cumulative_attention_scores = 0

    for _ in range(MAX_NEW_TOKENS):
        current = completed + active
        cumulative_attention_scores += attention_scores_per_step(len(current), model)
        input_ids = torch.tensor([current], device=device)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )["logits"]
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        if next_id == sent_id:
            completed += [token for token in active if token in structural] + [next_id]
            active = []
        else:
            active.append(next_id)

    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "cumulative_attention_scores": int(cumulative_attention_scores),
        "generated_tokens": len(generated),
    }


if __name__ == "__main__":
    prompt = [
        {   "content": "Hi there", 
            "role": "user"
        },
        {   "content": "Hello! How can I help you today?", 
            "role": "assistant"
        },
        {
            "content": "I'm looking for a beach resort for my next vacation. Can you recommend some popular ones?",
            "role": "user",
        },
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = run_full_attention(prompt, device)
    sparse = run_sparse_attention(prompt, device)
    print("HuggingFaceTB/SmolLM2-135M-Instruct model results:")
    print("Full text:", full["text"])
    print("Full generated tokens:", full["generated_tokens"])
    print("Full cumulative attention scores:", full["cumulative_attention_scores"])

    print("Sparse attention results:")
    print("Sparse text:", sparse["text"])
    print("Sparse generated tokens:", sparse["generated_tokens"])
    print("Sparse cumulative attention scores:", sparse["cumulative_attention_scores"])
