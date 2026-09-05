# Experiment Results Log

Date started: 2026-09-01
Project: Sentence-wise sparse attention with sentence-level KV cache

## Purpose
- Keep one simple, chronological record of experiments, outcomes, observations, and next decisions.
- Track what was tried, what failed, and why the next step was chosen.

## Current Big Goal
- Reduce KV memory and keep generation quality close to full attention.

## Decision Chain (High Level)
1. OOM during training in k8s -> moved to streaming/full-split iteration with memory-safe training settings.
2. Sparse inference gave empty output -> fixed decode path and aligned evaluation metrics with full attention path.
3. Aggressive pooled KV compression hurt quality -> tested keep-last-uncompressed and retrieval variants.
4. Layer-wise retrieval unstable -> moved to global cross-layer sentence selection.
5. Quality still degraded -> identified this as an off-distribution cache-surgery problem.
6. Previous compressed/retrieval generations often produced a locally meaningful sentence that was irrelevant to the conversation, followed by more sentences with the same failure mode.
7. New architecture -> train the local/global sentence-aware attention pattern directly instead of modifying a base model's cache only at inference time.

## Experiment Log

### E01 - Train Streaming/OOM Handling
- Objective: prevent pod OOM while still training on full dataset each epoch.
- Change:
  - stream dataset instead of loading all examples in memory.
  - preserve full split traversal when subset is all.
  - keep memory controls (checkpointing, cache off).
- Result:
  - training completed an epoch without pod OOM kill.
  - logs showed many steps with loss=0 and grad_norm=nan.
- Observation:
  - memory path improved, but optimization stability is still a concern.
- Decision:
  - continue architecture/inference debugging before deeper retraining.

### E02 - Sparse Inference Empty Output Fix
- Objective: fix sentence sparse model returning empty generation.
- Change:
  - decode generated token IDs directly in sparse path.
  - unify core metrics between full attention and sparse attention.
- Result:
  - sparse output no longer empty.
  - full vs sparse metrics are now comparable.
- Observation:
  - correctness improved, quality gap remained.
- Decision:
  - iterate on KV compression/retrieval design.

### E03 - Sentence Pool Cache Prototype
- Objective: compress sentence history while preserving useful context.
- Change:
  - chat-template span mapping for accurate sentence-to-token alignment.
  - keep last N completed sentences uncompressed.
- Result:
  - script runs reliably after tokenizer/cache compatibility fixes.
  - quality better than extreme compression but still inconsistent.
- Observation:
  - full replacement by pooled vectors loses important token detail.
- Decision:
  - test selective full-token retrieval from sentence pools.

### E04 - Sentence Representation Retrieval (Two-Pass)
- Objective: pass1 select relevant sentences, pass2 use full token K/V for selected sentences.
- Change:
  - implemented two-pass decode with sentence memory (pooled + full).
  - moved from per-layer selection to global aggregated selection shared across layers.
  - added controls: selection mode, repetition controls, forced sentence close fallback.
  - fixed technical issues: DynamicCache compatibility, device mismatch, position_ids/cache_position handling.
- Result:
  - pipeline executes and selection is now consistent across layers.
  - generation still frequently degenerates/repeats or becomes incoherent.
- Observation:
  - retrieval/cache reconstruction is still off-distribution for base LM behavior.
  - better selection policy alone is not sufficient.
- Decision:
  - move to trainable compressor experiment and/or distill retrieval path.

### E05 - Local/Global Sentence-Aware Attention Redesign
- Objective:
  - train the model with the same sentence-aware attention structure that is used during sparse inference.
  - retain sentence-level history while allowing the active sentence to use its word-level tokens.
- Change:
  - renamed the boundary token from `<SENT>` to `<|sent|>` and added it as a normal learned tokenizer/model vocabulary token.
  - removed the separate `sentence_vector` and `last_layer_local_proj` parameters and the embedding-copy initialization hack.
  - implemented two attention passes per decoder layer using the same attention weights:
    - local causal attention: a token attends only to earlier tokens in its own sentence segment, plus itself.
    - global causal attention: sentence tokens attend only to earlier sentence tokens, plus themselves.
  - the final decoder layer uses local attention only; all other layers use local attention followed by global attention.
  - applied the same attention structure in training and inference to remove the previous train/inference mismatch.
  - enabled gradients for the complete model so the normal `<|sent|>` embedding and all attention/MLP weights can adapt to the new pattern.
  - inference keeps completed sentence-marker tokens and the active sentence's marker/word tokens. When the model predicts a new `<|sent|>`, word tokens from the closed sentence are removed from the active sequence.
- Training configuration and result:
  - model: `HuggingFaceTB/SmolLM2-135M-Instruct`.
  - one epoch over the full dataset, 7,193 steps.
  - maximum sequence length: 1,024; batch size: 8; gradient accumulation: 8.
  - learning rate: 0.0002; weight decay: 0.01; fp32; gradient checkpointing disabled in this run.
  - runtime: approximately 24 hours 51 minutes.
  - final reported training loss: 15.81.
- Result:
  - training completed and the new checkpoint was written to `sentence-local-global-smollm2-135m`.
  - the custom model forward pass was previously smoke-tested with finite logits/loss and no NaNs, including padded input.
  - the subsequent `src/inference.py` run exited with status 1; an end-to-end generation result has not yet been recorded.
- Observation:
  - the redesign changes the model's learned computation rather than relying only on post-hoc KV-cache surgery, but the high training loss and failed inference validation mean quality and checkpoint compatibility still need to be checked.
  - causally, a `<|sent|>` token appears before the words of its sentence and therefore cannot receive information from those future words through local attention. Cross-sentence information currently travels through the sentence-token chain; this is an important limitation to evaluate.
- Decision:
  - debug and validate inference before starting another long training run.

## Quality Observations
- In the earlier pooling and retrieval experiments, the generated sentence was often grammatically valid and could express a correct sentence-level meaning, but it was irrelevant to the active conversation context.
- This was not isolated to one sentence: each newly generated sentence frequently showed the same context-independent behavior, causing repetition, topic drift, or incoherent multi-sentence responses.
- The likely cause is that the base LM was trained for dense token-level context, while inference replaced that context with pooled or reconstructed KV states. The resulting hidden states were off-distribution even when the retrieved sentence itself appeared relevant.
- Therefore, selecting a more relevant sentence alone did not solve the problem; the model must be trained to consume the compressed/sentence-level representation in the same way it will be used during generation.

## Baseline Snapshot
- Full attention baseline in inference script can generate coherent beach-resort style answer.
- Retrieval/compressed variants currently trade memory for notable quality degradation.
- The new local/global model has completed training, but no quality comparison should be reported until `src/inference.py` runs successfully against the new checkpoint.

## New Experiment Proposal - Sentence Compressor
Status: deferred until the local/global redesign is validated

### Hypothesis
- A trainable sentence compressor can encode sentence token sequences into LM-compatible sentence vectors.
- If trained against next-token objective (with frozen base LM), compressed context may preserve more semantics than naive mean/max pooling.

### Minimal First Version
- Keep base SmolLM frozen.
- Train only:
  - sentence compressor module,
  - sentence vector bridge/projection (if needed),
  - minimal sparse control parameters already proven trainable.
- Input pipeline:
  - sentence word tokens -> compressor vector,
  - plus recent uncompressed tokens,
  - predict next token.

### Success Criteria
- Better response quality than pooled/retrieval heuristic on fixed prompts.
- Lower repetition/degeneration rate.
- Memory lower than full attention baseline.

## Entry Template (Copy for Each New Trial)
### EXX - Name
- Objective:
- Change:
- Command:
- Key config:
- Result:
- Observation:
- Decision:



## Next Steps


1. since this is unidirectional attention, keep the <|sent|> token at the end of each sentence. so that next words/sent attent to the sent that has the sentence information. keeping at beginning will make the sent to be attended before its words, which may limit information flow.

2. instead of local and global make it one single attention pass. where word attents to previous words and previous sentence representations.

