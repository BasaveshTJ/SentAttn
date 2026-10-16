import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


class SentenceSparseSmolLM2WithKVCache(LlamaForCausalLM):
    """Causal attention with a cache containing only useful tokens."""

    def __init__(self, config):
        super().__init__(config)
        self.sentence_token_id = getattr(config, "sentence_token_id", -1)
        self.structural_token_ids = set(getattr(config, "structural_token_ids", []))

    def set_sentence_token_id(self, token_id):
        self.sentence_token_id = int(token_id)
        self.config.sentence_token_id = int(token_id)

    def set_structural_token_ids(self, token_ids):
        self.structural_token_ids = set(map(int, token_ids))
        self.config.structural_token_ids = list(self.structural_token_ids)

    def allowed_mask(self, input_ids):
        # Same segment-restricted sparse mask as the original sparse model.
        # Needed only for the prefill pass, where multiple prompt tokens from
        # different segments are embedded in a single call.
        bsz, length = input_ids.shape
        device = input_ids.device
        sent = input_ids.eq(self.sentence_token_id)
        structural = torch.zeros_like(sent)
        for token_id in self.structural_token_ids:
            structural |= input_ids.eq(token_id)
        word = ~(sent | structural)
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
        return allowed

    def _prune_cache(self, cache, token_ids):
        sent = token_ids.eq(self.sentence_token_id)
        keep = sent.clone()
        for token_id in self.structural_token_ids:
            keep |= token_ids.eq(token_id)

        # Keep the active sentence; completed sentence words are removed.
        if sent.any():
            last = sent.long().cumsum(1).argmax(1)
            keep[:, last[0] + 1:] = True
        else:
            keep[:] = True

        indices = keep[0].nonzero(as_tuple=True)[0]
        for layer in cache.layers:
            layer.keys = layer.keys.index_select(-2, indices)
            layer.values = layer.values.index_select(-2, indices)
        cache._token_ids = token_ids.index_select(1, indices)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        cache = past_key_values if use_cache else None
        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        cached_ids = getattr(cache, "_token_ids", None) if cache is not None else None
        if cached_ids is None:
            full_ids = input_ids
            cache_length = 0
        else:
            full_ids = torch.cat((cached_ids, input_ids), dim=1)
            cache_length = cached_ids.shape[1]

        query_length = input_ids.shape[1]
        if position_ids is None:
            start = getattr(cache, "_next_position", 0)
            position_ids = torch.arange(
                start, start + query_length, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self.model.embed_tokens(input_ids)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        key_length = cache_length + query_length

        if cache_length == 0 and query_length > 1:
            # Prefill: tokens from different segments are processed together,
            # so the segment-restricted sparse mask must still be applied.
            allowed = self.allowed_mask(full_ids)
        else:
            # Decode: the cache was already pruned to exactly the tokens a new
            # query is allowed to see, so plain causal attention is enough.
            allowed = torch.ones(
                query_length, key_length, device=input_ids.device, dtype=torch.bool
            )[None]
        bias = torch.zeros(
            (1, 1, query_length, key_length),
            device=input_ids.device,
            dtype=hidden_states.dtype,
        ).masked_fill(~allowed[:, None, -query_length:, :], torch.finfo(hidden_states.dtype).min)

        for layer in self.model.layers:
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            attn_out, _ = layer.self_attn(
                hidden_states=normed,
                attention_mask=bias,
                position_embeddings=position_embeddings,
                past_key_values=cache,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            normed = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(normed)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if cache is not None:
            cache._next_position = getattr(cache, "_next_position", 0) + query_length
            self._prune_cache(cache, full_ids)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=cache
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # `generate()` pre-creates an empty cache object before the first call,
        # so check for actually cached tokens rather than `past_key_values is
        # not None` to decide whether this is the prefill step.
        if getattr(past_key_values, "_token_ids", None) is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
