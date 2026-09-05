import os
import math

import torch
from datasets import load_dataset, load_dataset_builder
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from model import SentenceSparseSmolLM2ForCausalLM
from sentence import SENT_TOKEN, add_sentence_tokens, add_sentence_tokens_from_messages


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
OUTPUT_DIR = "./sentence-local-global-smollm2-135m-all"
RESUME_FROM_CHECKPOINT = None
SUBSET = "all"
# SUBSET = "everyday-conversations"
NUM_EPOCHS = 1
MAX_LENGTH = 1024
TRAIN_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.001
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
USE_GRAD_CHECKPOINTING = False
SEED = 42
DATALOADER_NUM_WORKERS = 0
DATALOADER_PIN_MEMORY = False


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
    def __init__(self, tokenizer, expected_length=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.expected_length = expected_length

    def __iter__(self):
        ds = load_dataset(DATASET_ID, split="train", streaming=True)
        for row in ds:
            if "messages" not in row:
                continue
            ex = encode_example(row["messages"], self.tokenizer)
            yield {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}

    def __len__(self):
        if self.expected_length is None:
            raise TypeError("Streaming dataset length is unknown")
        return self.expected_length


def build_train_dataset(tokenizer):
    if SUBSET == "all":
        train_size = load_dataset_builder(DATASET_ID).info.splits["train"].num_examples
        return StreamTokenizedDataset(tokenizer=tokenizer, expected_length=train_size)
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.filter(lambda x: x["source"] == SUBSET)
    cols = ds.column_names
    ds = ds.map(lambda x: encode_example(x["messages"], tokenizer), remove_columns=cols)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def train():
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training setup")
    use_bf16 = False
    use_fp16 = False

    tokenizer = ensure_tokenizer(MODEL_ID)
    model = SentenceSparseSmolLM2ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    sent_token_id = tokenizer.convert_tokens_to_ids(SENT_TOKEN)
    model.set_sentence_token_id(sent_token_id)
    for p in model.parameters():
        p.requires_grad = True
    model.config.use_cache = False
    if USE_GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Model:", MODEL_ID)
    print("Subset:", SUBSET)
    print("Sentence token id:", sent_token_id)
    print("Trainable params:", trainable, "/", total)
    print("Max length:", MAX_LENGTH)
    print("Train batch size:", TRAIN_BATCH_SIZE)
    print("Grad accumulation steps:", GRAD_ACCUM_STEPS)
    print("Learning rate:", LEARNING_RATE)
    print("Warmup ratio:", WARMUP_RATIO)
    print("Weight decay:", WEIGHT_DECAY)
    print("Max grad norm:", MAX_GRAD_NORM)
    print("Gradient checkpointing:", USE_GRAD_CHECKPOINTING)
    print("Precision mode: fp32")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dataset = build_train_dataset(tokenizer)
    train_examples = len(train_dataset)
    micro_batches_per_epoch = math.ceil(train_examples / TRAIN_BATCH_SIZE)
    update_steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / GRAD_ACCUM_STEPS))
    total_update_steps = max(1, update_steps_per_epoch * NUM_EPOCHS)
    warmup_steps = max(1, int(total_update_steps * WARMUP_RATIO))
    print("Warmup steps:", warmup_steps)

    args_kwargs = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "gradient_checkpointing": USE_GRAD_CHECKPOINTING,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "constant",
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "logging_steps": 50,
        "save_total_limit": 5,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": SEED,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": DATALOADER_PIN_MEMORY,
    }
    args_kwargs.update({"num_train_epochs": NUM_EPOCHS, "save_strategy": "epoch"})

    training_args = TrainingArguments(**args_kwargs)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=default_data_collator)

    if RESUME_FROM_CHECKPOINT is not None:
        print("Resuming from checkpoint:", RESUME_FROM_CHECKPOINT)
    else:
        print("Starting training from scratch")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training completed")


if __name__ == "__main__":
    train()


# python -m spacy download en_core_web_sm
