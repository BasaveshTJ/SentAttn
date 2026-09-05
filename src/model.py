from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.utils.generic import ModelOutput


@dataclass
class SentenceSparseOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class SentenceSparseSmolLM2ForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.sentence_token_id = getattr(config, "sentence_token_id", -1)

    # Set sentence token id
    def set_sentence_token_id(self, sentence_token_id):
        self.sentence_token_id = int(sentence_token_id)
        self.config.sentence_token_id = int(sentence_token_id)

    # Get sentence segment ids: <|sent|> starts a new segment that includes itself
    # and every word token up to (excluding) the next <|sent|>.
    def _sentence_ids(self, input_ids):
        if self.sentence_token_id < 0:
            raise ValueError("sentence_token_id is not set")
        sent_hits = (input_ids == self.sentence_token_id).long()
        sent_ids = torch.cumsum(sent_hits, dim=1) - 1
        return sent_ids.clamp_min(0)

    # Causal + padding-safe additive bias from a boolean "allowed" matrix
    def _bias_from_allowed(self, allowed, attention_mask_2d, dtype):
        bsz, seq_len, _ = allowed.shape
        device = allowed.device
        if attention_mask_2d is not None:
            valid = attention_mask_2d.bool()
            allowed = allowed & valid.unsqueeze(1)
        eye = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
        allowed = allowed | eye  # always allow self-attention so no row is fully masked
        mask = torch.full((bsz, 1, seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        return mask.masked_fill(allowed.unsqueeze(1), 0.0)

    # Local bias: causal attention restricted to tokens within the same sentence segment
    def _build_local_bias(self, input_ids, attention_mask_2d, dtype):
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        sent_ids = self._sentence_ids(input_ids)
        idx = torch.arange(seq_len, device=device)
        causal = (idx.unsqueeze(0) <= idx.unsqueeze(1)).unsqueeze(0).expand(bsz, -1, -1)
        same_sentence = sent_ids.unsqueeze(2) == sent_ids.unsqueeze(1)
        allowed = causal & same_sentence
        return self._bias_from_allowed(allowed, attention_mask_2d, dtype)

    # Global bias: causal attention where only <|sent|> tokens attend to earlier <|sent|> tokens
    def _build_global_bias(self, input_ids, attention_mask_2d, dtype):
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        is_sent = input_ids == self.sentence_token_id
        idx = torch.arange(seq_len, device=device)
        causal = (idx.unsqueeze(0) <= idx.unsqueeze(1)).unsqueeze(0).expand(bsz, -1, -1)
        key_is_sent = is_sent.unsqueeze(1).expand(-1, seq_len, -1)
        query_is_sent = is_sent.unsqueeze(2).expand(-1, -1, seq_len)
        allowed = causal & key_is_sent & query_is_sent
        return self._bias_from_allowed(allowed, attention_mask_2d, dtype)

    # Custom local+global sparse forward
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self.model.embed_tokens(input_ids)
        pos_emb = self.model.rotary_emb(hidden_states, position_ids)
        local_bias = self._build_local_bias(input_ids, attention_mask, hidden_states.dtype)
        global_bias = self._build_global_bias(input_ids, attention_mask, hidden_states.dtype)

        num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            # Pass 1 (all layers): local, same-sentence causal attention.
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            attn_out, _ = layer.self_attn(hidden_states=normed, attention_mask=local_bias, position_ids=position_ids, position_embeddings=pos_emb)
            hidden_states = residual + attn_out

            # Pass 2 (all layers except the last): global, <|sent|>-to-<|sent|> causal attention.
            # Reuses the same self_attn weights as the local pass above.
            if i != num_layers - 1:
                residual = hidden_states
                normed = layer.input_layernorm(hidden_states)
                attn_out, _ = layer.self_attn(hidden_states=normed, attention_mask=global_bias, position_ids=position_ids, position_embeddings=pos_emb)
                hidden_states = residual + attn_out

            residual = hidden_states
            normed = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(normed)

        hidden_states = self.model.norm(hidden_states)
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

        return SentenceSparseOutput(loss=loss, logits=logits)



