from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.utils.generic import ModelOutput


@dataclass
class SentenceSparseOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    sparse_mask_penultimate: torch.Tensor | None = None
    sparse_mask_last: torch.Tensor | None = None


class SentenceSparseSmolLM2ForCausalLM(LlamaForCausalLM):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model._reset_sparse_params_if_needed()
        return model

    def __init__(self, config):
        super().__init__(config)
        self.sentence_token_id = getattr(config, "sentence_token_id", -1)
        self.sentence_vector = nn.Parameter(torch.zeros(config.hidden_size))
        self.last_layer_local_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_init()
        with torch.no_grad():
            self.last_layer_local_proj.weight.zero_()

    # Keep new params finite and stable
    def _reset_sparse_params_if_needed(self):
        if getattr(self.sentence_vector, "is_meta", False):
            return
        with torch.no_grad():
            if not torch.isfinite(self.sentence_vector).all():
                self.sentence_vector.zero_()
            if not torch.isfinite(self.last_layer_local_proj.weight).all():
                self.last_layer_local_proj.weight.zero_()

    # Set sentence token id
    def set_sentence_token_id(self, sentence_token_id):
        self.sentence_token_id = int(sentence_token_id)
        self.config.sentence_token_id = int(sentence_token_id)

    # Freeze all except experiment params
    def freeze_except_sparse_params(self):
        allowed = {"sentence_vector", "last_layer_local_proj.weight"}
        for name, p in self.named_parameters():
            p.requires_grad = name in allowed

    # Get sentence segment ids
    def _sentence_ids(self, input_ids):
        if self.sentence_token_id < 0:
            raise ValueError("sentence_token_id is not set")
        sent_hits = (input_ids == self.sentence_token_id).long()
        sent_ids = torch.cumsum(sent_hits, dim=1) - 1
        return sent_ids.clamp_min(0)

    # Build sparse mask for one layer type
    def _build_sparse_mask(self, input_ids, attention_mask_2d=None, last_layer=False, focus_last_sentence=False, dtype=torch.float32):
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        sent_ids = self._sentence_ids(input_ids)
        is_sent = input_ids == self.sentence_token_id

        idx = torch.arange(seq_len, device=device)
        causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
        causal = causal.unsqueeze(0).expand(bsz, -1, -1)

        q_sid = sent_ids.unsqueeze(2)
        k_sid = sent_ids.unsqueeze(1)
        same_sentence = q_sid == k_sid
        key_is_sentence = is_sent.unsqueeze(1).expand(-1, seq_len, -1)
        query_is_sentence = is_sent.unsqueeze(2).expand(-1, -1, seq_len)

        if last_layer and focus_last_sentence:
            last_sid = sent_ids.max(dim=1, keepdim=True).values
            q_last = q_sid == last_sid.unsqueeze(2)
            k_last = k_sid == last_sid.unsqueeze(1)
            self_only = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
            allowed = (causal & q_last & k_last) | ((~q_last) & self_only)
        else:
            sentence_query_allowed = key_is_sentence
            token_query_allowed = same_sentence | key_is_sentence
            allowed = torch.where(query_is_sentence, sentence_query_allowed, token_query_allowed)
            allowed = allowed & causal

        if attention_mask_2d is not None:
            valid = attention_mask_2d.bool()
            k_valid = valid.unsqueeze(1)
            allowed = allowed & k_valid
            pad_q = (~valid).unsqueeze(2)
            pad_self = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
            allowed = allowed | (pad_q & pad_self)

        mask = torch.full((bsz, 1, seq_len, seq_len), -1e4, device=device, dtype=dtype)
        mask = mask.masked_fill(allowed.unsqueeze(1), 0.0)
        return mask

    # Masked sentence vector injection
    def _inject_sentence_vector(self, input_ids):
        embeds = self.model.embed_tokens(input_ids)
        sent_mask = input_ids == self.sentence_token_id
        if sent_mask.any():
            embeds = torch.where(sent_mask.unsqueeze(-1), self.sentence_vector.view(1, 1, -1), embeds)
        return embeds

    # Count allowed sparse attention edges
    def count_allowed_edges(self, input_ids, attention_mask=None, last_layer=False):
        with torch.no_grad():
            mask = self._build_sparse_mask(input_ids, attention_mask_2d=attention_mask, last_layer=last_layer)
            return (mask == 0).sum().item()

    # Custom sparse forward
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, use_cache=False, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self._inject_sentence_vector(input_ids)
        pos_emb = self.model.rotary_emb(hidden_states, position_ids)
        sparse_mask_penultimate = self._build_sparse_mask(input_ids, attention_mask_2d=attention_mask, last_layer=False, dtype=hidden_states.dtype)
        # Training: final layer runs for all sentences.
        # Inference: final layer focuses only on the last sentence.
        sparse_mask_last = self._build_sparse_mask(
            input_ids,
            attention_mask_2d=attention_mask,
            last_layer=True,
            focus_last_sentence=not self.training,
            dtype=hidden_states.dtype,
        )

        for i, layer in enumerate(self.model.layers):
            layer_mask = sparse_mask_last if i == len(self.model.layers) - 1 else sparse_mask_penultimate
            hidden_states = layer(hidden_states=hidden_states, attention_mask=layer_mask, position_ids=position_ids, position_embeddings=pos_emb, use_cache=use_cache)

        hidden_states = self.model.norm(hidden_states)
        sent_ids = self._sentence_ids(input_ids)
        last_sid = sent_ids.max(dim=1, keepdim=True).values
        last_token_mask = (sent_ids == last_sid).unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = hidden_states + self.last_layer_local_proj(hidden_states * last_token_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid_count = (shift_labels != -100).sum()
            if valid_count.item() == 0:
                loss = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
            else:
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return SentenceSparseOutput(loss=loss, logits=logits, sparse_mask_penultimate=sparse_mask_penultimate, sparse_mask_last=sparse_mask_last)
