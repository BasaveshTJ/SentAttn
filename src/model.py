import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM


class SentenceSparseSmolLM2ForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.sentence_token_id = getattr(config, "sentence_token_id", -1)
        self.structural_token_ids = set(getattr(config, "structural_token_ids", []))

    def set_sentence_token_id(self, sentence_token_id):
        self.sentence_token_id = int(sentence_token_id)
        self.config.sentence_token_id = int(sentence_token_id)

    def set_structural_token_ids(self, token_ids):
        self.structural_token_ids = set(map(int, token_ids))
        self.config.structural_token_ids = list(self.structural_token_ids)

    def allowed_mask(self, input_ids, attention_mask):
        bsz, length = input_ids.shape
        device = input_ids.device
        sent = input_ids.eq(self.sentence_token_id)
        structural = torch.zeros_like(sent)
        for token_id in self.structural_token_ids:
            structural |= input_ids.eq(token_id)
        word = ~(sent | structural)
        # A sentence marker belongs to the sentence immediately before it.
        segment = torch.cumsum(sent.long(), 1) - sent.long()
        query = torch.arange(length, device=device)[None, :, None]
        key = torch.arange(length, device=device)[None, None, :]
        causal = key <= query
        same_segment = segment[:, :, None].eq(segment[:, None, :])
        previous_sent = sent[:, None, :] & (key < query)
        previous_structural = structural[:, None, :] & causal
        word_keys = word[:, None, :] & causal

        allowed_words = same_segment & word_keys
        allowed = torch.where(word[:, :, None], allowed_words | previous_sent | previous_structural, False)
        allowed = torch.where(sent[:, :, None], allowed_words | previous_sent | previous_structural, allowed)
        allowed = torch.where(structural[:, :, None], previous_sent | previous_structural, allowed)
        allowed |= torch.eye(length, device=device, dtype=torch.bool)[None]
        if attention_mask is not None:
            allowed &= attention_mask.bool()[:, None, :]
        return allowed

    def _attention_bias(self, input_ids, attention_mask, dtype):
        bsz, length = input_ids.shape
        allowed = self.allowed_mask(input_ids, attention_mask)
        bias = torch.full((bsz, 1, length, length), torch.finfo(dtype).min, device=input_ids.device, dtype=dtype)
        return bias.masked_fill(allowed[:, None], 0.0)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self.model.embed_tokens(input_ids)
        pos_emb = self.model.rotary_emb(hidden_states, position_ids)
        bias = self._attention_bias(input_ids, attention_mask, hidden_states.dtype)

        for layer in self.model.layers:
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            attn_out, _ = layer.self_attn(hidden_states=normed, attention_mask=bias, position_ids=position_ids, position_embeddings=pos_emb)
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

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}