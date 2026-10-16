import torch

from inference import (
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    load_full_model,
    load_sparse_model,
    track_generation_stats,
)
from sentence import add_sentence_tokens_to_messages, add_system_sentence_token


def encode_full(tokenizer, messages, device):
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (encoded.input_ids if hasattr(encoded, "input_ids") else encoded).to(device)
    return input_ids, torch.ones_like(input_ids)


def encode_sparse(tokenizer, messages, device):
    text = tokenizer.apply_chat_template(
        add_sentence_tokens_to_messages(messages), tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor([tokenizer(add_system_sentence_token(text)).input_ids], device=device)
    return input_ids, torch.ones_like(input_ids)


def clean_reply(reply):
    return reply.replace("<|im_end|>", "").replace("<|sent|>", " ").strip()


def generate_reply(model, tokenizer, input_ids, attention_mask):
    stats, handle = track_generation_stats(model)
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
    reply = tokenizer.decode(generated, skip_special_tokens=False).strip()
    print(f"  tokens: {len(generated)}")
    print(f"  cache kv tokens (max/last): {max(stats['cache_sizes'], default=0)}/{stats['cache_sizes'][-1] if stats['cache_sizes'] else 0}")
    print(f"  attention scores (cumulative/last): {sum(stats['attention_scores'])}/{stats['attention_scores'][-1] if stats['attention_scores'] else 0}")
    return reply


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_model, full_tokenizer = load_full_model(device)
    sparse_model, sparse_tokenizer = load_sparse_model(device)
    full_messages, sparse_messages = [], []

    print("Chat with both models. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        full_messages.append({"role": "user", "content": user_input})
        sparse_messages.append({"role": "user", "content": user_input})

        print("\nFull attention:")
        input_ids, attention_mask = encode_full(full_tokenizer, full_messages, device)
        full_reply = generate_reply(full_model, full_tokenizer, input_ids, attention_mask)
        print(f"  reply: {full_reply}\n")
        full_messages.append({"role": "assistant", "content": clean_reply(full_reply)})

        print("Sparse attention:")
        input_ids, attention_mask = encode_sparse(sparse_tokenizer, sparse_messages, device)
        sparse_reply = generate_reply(
            sparse_model, sparse_tokenizer, input_ids, attention_mask
        )
        print(f"  reply: {sparse_reply}\n")
        sparse_messages.append({"role": "assistant", "content": clean_reply(sparse_reply)})


if __name__ == "__main__":
    main()
