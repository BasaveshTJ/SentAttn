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
6. Next idea -> train a sentence compressor jointly with frozen base LM and small trainable modules.

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

## Baseline Snapshot
- Full attention baseline in inference script can generate coherent beach-resort style answer.
- Retrieval/compressed variants currently trade memory for notable quality degradation.

## New Experiment Proposal - Sentence Compressor
Status: planned

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
