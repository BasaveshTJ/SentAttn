import spacy

SENT_TOKEN = "<|sent|>"
_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
        if "parser" not in _NLP.pipe_names and "senter" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP


def add_sentence_tokens(text, sent_token=SENT_TOKEN):
    doc = get_nlp()(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sentences:
        return text.strip()
    return " ".join(f"{s} {sent_token}" for s in sentences)


def add_sentence_tokens_to_messages(messages, sent_token=SENT_TOKEN):
    output = []
    for message in messages:
        updated_message = dict(message)
        updated_message["content"] = add_sentence_tokens(updated_message.get("content", ""), sent_token)
        output.append(updated_message)
    return output


def add_system_sentence_token(text, sent_token=SENT_TOKEN):
    return text.replace("<|im_end|>", f"{sent_token}<|im_end|>", 1)


def sample_run():
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    tokenizer.add_special_tokens({"additional_special_tokens": [SENT_TOKEN]})
    messages = add_sentence_tokens_to_messages(prompt)
    print("Messages after adding sentence tokens:")
    for message in messages:
        print(message)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = add_system_sentence_token(text)
    ids = tokenizer(text).input_ids
    print("Token IDs after tokenization:")
    print(tokenizer.decode(ids, skip_special_tokens=False))
    print("Tokens after conversion from IDs:")
    print(tokenizer.convert_ids_to_tokens(ids))

    import torch
    from model import SentenceSparseSmolLM2ForCausalLM

    model = SentenceSparseSmolLM2ForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.set_sentence_token_id(tokenizer.convert_tokens_to_ids(SENT_TOKEN))
    model.set_structural_token_ids([
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<|im_start|>", "<|im_end|>", "system", "user", "assistant")
    ])
    input_ids = torch.tensor([ids])
    bias = model._attention_bias(input_ids, torch.ones_like(input_ids), model.dtype)
    attended = bias[0, 0, -1].eq(0).nonzero(as_tuple=True)[0].tolist()
    attended_ids = [ids[i] for i in attended]
    print("Last token:", tokenizer.convert_ids_to_tokens([ids[-1]])[0])
    print("Attended token positions:", attended)
    print("Attended tokens:", tokenizer.convert_ids_to_tokens(attended_ids))
    print("Attended text:", tokenizer.decode(attended_ids, skip_special_tokens=False))
    return prompt


if __name__ == "__main__":
    sample_run()
