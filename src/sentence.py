import spacy

SENT_TOKEN = "<|sent|>"
_NLP = None


# Load spacy once
def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
        if "parser" not in _NLP.pipe_names and "senter" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP

# Split text into sentences
def split_sentences(text):
    doc = get_nlp()(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


# Split each chat message separately
def split_chat_sentences(messages):
    out = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        parts = split_sentences(content)
        if not parts:
            clean = content.strip()
            if clean:
                out.append(f"{role}: {clean}")
            continue
        out.append(f"{role}: {parts[0]}")
        out.extend(parts[1:])
    return out


# Add sentence token before each sentence
def add_sentence_tokens(text, sent_token=SENT_TOKEN):
    sentences = split_sentences(text)
    if not sentences:
        return f"{sent_token} {text.strip()}".strip()
    return " ".join([f"{sent_token} {s}" for s in sentences])


# Add sentence token on chat sentences
def add_sentence_tokens_from_messages(messages, sent_token=SENT_TOKEN):
    sentences = split_chat_sentences(messages)
    if not sentences:
        return sent_token
    return " ".join([f"{sent_token} {s}" for s in sentences])


# Sample sentence split test
def run_sample(prompt):
    sample_text = " ".join([f"{m['role'].capitalize()}: {m['content']}" for m in prompt])
    print("Sample text:", sample_text)
    print("Message-level sentences:", split_chat_sentences(prompt))
    print("With sentence tokens:", add_sentence_tokens_from_messages(prompt))


if __name__ == "__main__":
    prompt =  [
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

    run_sample(prompt)
