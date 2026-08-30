import os

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from model import SentenceSparseSmolLM2ForCausalLM
from sentence import SENT_TOKEN, add_sentence_tokens, add_sentence_tokens_from_messages


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
OUTPUT_DIR = "./sentence-sparse-smollm2-135m"
# SUBSET = "all"
SUBSET = "everyday-conversations" # for testing, using small dataset
NUM_EPOCHS = 5
MAX_LENGTH = 1024
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
STREAM_SAMPLES_PER_EPOCH = 20000
SEED = 42


def ensure_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if SENT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [SENT_TOKEN]})
    tokenizer.padding_side = "right"
    return tokenizer


def encode_example(messages, tokenizer):
    text = add_sentence_tokens_from_messages(messages) if isinstance(messages, list) else add_sentence_tokens(tokenizer.apply_chat_template(messages, tokenize=False))
    tok = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    labels = [tid if m == 1 else -100 for tid, m in zip(input_ids, attention_mask)]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Streaming dataset wrapper
class StreamTokenizedDataset(IterableDataset):
    def __init__(self, tokenizer, sample_limit):
        super().__init__()
        self.tokenizer = tokenizer
        self.sample_limit = sample_limit

    def __iter__(self):
        ds = load_dataset(DATASET_ID, split="train", streaming=True)
        count = 0
        for row in ds:
            if "messages" not in row:
                continue
            ex = encode_example(row["messages"], self.tokenizer)
            yield {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}
            count += 1
            if self.sample_limit is not None and count >= self.sample_limit:
                break


def build_train_dataset(tokenizer):
    if SUBSET == "all":
        return StreamTokenizedDataset(tokenizer=tokenizer, sample_limit=STREAM_SAMPLES_PER_EPOCH)
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.filter(lambda x: x["source"] == SUBSET)
    cols = ds.column_names
    ds = ds.map(lambda x: encode_example(x["messages"], tokenizer), remove_columns=cols)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def train():
    torch.manual_seed(SEED)
    tokenizer = ensure_tokenizer(MODEL_ID)
    model = SentenceSparseSmolLM2ForCausalLM.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    sent_token_id = tokenizer.convert_tokens_to_ids(SENT_TOKEN)
    with torch.no_grad():
        ref_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        model.model.embed_tokens.weight[sent_token_id].copy_(model.model.embed_tokens.weight[ref_id])
        model.lm_head.weight[sent_token_id].copy_(model.lm_head.weight[ref_id])
    model.set_sentence_token_id(sent_token_id)
    model.freeze_except_sparse_params()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Model:", MODEL_ID)
    print("Subset:", SUBSET)
    print("Sentence token id:", sent_token_id)
    print("Trainable params:", trainable, "/", total)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dataset = build_train_dataset(tokenizer)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    args_kwargs = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 50,
        "save_total_limit": 5,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": SEED,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "dataloader_num_workers": 0,
    }
    if SUBSET == "all":
        steps_per_epoch = max(1, STREAM_SAMPLES_PER_EPOCH // (TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))
        args_kwargs.update({"max_steps": steps_per_epoch * NUM_EPOCHS, "save_strategy": "steps", "save_steps": steps_per_epoch})
    else:
        args_kwargs.update({"num_train_epochs": NUM_EPOCHS, "save_strategy": "epoch"})

    training_args = TrainingArguments(**args_kwargs)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=default_data_collator)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training completed")


if __name__ == "__main__":
    train()
