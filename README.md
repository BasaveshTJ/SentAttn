# SentAttn

SentAttn explores sentence-aware sparse attention for decoder-only language models.
SentAttn explores sentence-aware sparse attention for decoder-only language models.
The core idea is to treat each sentence boundary token as a sentence context collector, so later tokens can read compact sentence context instead of all past word tokens.
## Overview

- Base model: HuggingFaceTB/SmolLM2-135M-Instruct
- Data: HuggingFaceTB/smol-smoltalk (everyday-conversations and full-data runs)
- Main mechanism:
  - Insert `<|sent|>` at sentence boundaries.
  - Keep dense word-level attention inside the current sentence.
  - Let tokens attend to prior sentence markers as compact history.
- Goal: reduce attention/KV cost while keeping response quality usable.

### Current Direction 

The current direction is a single-pass sentence-aware mask (not two-pass local/global blocks).

- Word token attention: same-sentence previous words + prior sentence markers + structural tokens.
- Sentence marker attention: same-sentence words + prior sentence markers + structural tokens.
- Structural token attention: causal structural/sentence-marker context.

This is the setup implemented in [src/model.py](src/model.py), trained from [src/train.py](src/train.py), and evaluated from [src/inference.py](src/inference.py).

### Observed Outcomes

- Coherent generations on small prompt checks.
- Large drop in attended positions versus full causal masking in sample runs.
- Sentence context collectors often helped preserve sentence-level information in their hidden states.
- Compression-heavy pooling/retrieval variants degraded quality.
- tested only on context with few sentences, this is not tested on longer context and can't conclude its effectiveness there.
- Behavior is sensitive to sentence segmentation quality.

Detailed runs, prompt-by-prompt outputs and observeations are in [src/results.md](results.md).

### Next Steps

1. Evaluate sentence context collector on longer contexts.
2. Investigate whether other tokens could serve as context-capturing tokens if trained appropriately.

The `<|sent|>` token was also a token similar to any other token in the vocabulary, but it was able to capture the sentence-level context. This was due to the training with custom masking strategies that emphasized sentence boundaries.
so could this means any other tokens could also serve as context-capturing tokens and also serve as vocabulary tokens if trained appropriately? and capture fine-grained contextual information within phrases rather than compressing the whole sentence into one representation.

The next experiment will be around this idea to test whether existing tokens can serve as context-capturing tokens instead of introducing a dedicated `<|sent|>` token. Can a token at the end of the phrase could summarize the preceding phrase, reducing the number of tokens that later positions need to attend to without compressing an entire sentence into one representation.

For example, in the sentence "The Jamaica market is generally considered a more affordable option, with better deals on food and services," candidate phrase-ending tokens could be:

- `market` for "The Jamaica market"
- `considered` for "is generally considered"
- `option` for "a more affordable option"
- `deals` for "better deals"
- `food` for "on food"
- `services` for "and services"

If each word were represented by one token, this example would contain approximately 18 tokens and could be represented by six context-capturing tokens and this is a significant reduction in the number of tokens that later positions need to attend to.

The context-capturing token should appear at the end of the phrase it represents, and the attention mask should allow later tokens to attend primarily to these tokens rather than to every token in the phrase.The main challenge is identifying phrase boundaries reliably.

