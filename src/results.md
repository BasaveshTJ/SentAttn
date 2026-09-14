# SentAttn Results

## Purpose

Track observations while testing sentence-aware sparse attention, where `<|sent|>` acts as a sentence context collector.
This document is intentionally observational, not a final conclusion.

## Setup Snapshot

- Base model: HuggingFaceTB/SmolLM2-135M-Instruct
- Main dataset: HuggingFaceTB/smol-smoltalk
- Typical generation cap: 100 tokens
- Sentence context collector insertion at sentence boundaries: spaCy (`en_core_web_sm`)
- Structural tokens used in masking: `<|im_start|>`, `<|im_end|>`, `system`, `user`, `assistant`

## Experiments

### E01: Sentence Pool Replacement

Method:

- Pool all token's K/V from each sentence into one representation.

Observation:

- Quality dropped noticeably.
- Keeping a few recent sentences uncompressed helped a little but did not remove the quality gap.

### E02: Sentence-Level Retrieval from Pooled Keys

Method:

- Keep pooled tokens of a sentence inot one representations in cache.
- Retrieve top relevant sentences and expand to full tokens for those only.

Observation:

- Simple pooled keys were often weak as retrieval anchors.
- Top sentence selection was unstable for semantically close options.

### E03: Two-Pass Local/Global Attention

Method:

- Add `<|sent|>` as sentence representation token.
- Use a local pass and a global pass with shared QKV weights in every penultimate layer.
- in the final layer perform only the local attention.

Observation:

- Optimization did not converge reliably.
- Added architectural complexity and overhead without clear quality gain.
- added unnecessary complexity without clear benefit.

### E04: Single-Pass Sentence-Aware Mask (Primary Direction)

Method:

- Add new special token called `<|sent|>` at sentence boundaries.
- Use one attention pass per decoder layer.
- Train and infer with the same sentence-aware mask.

Mask behavior:

- Word tokens attend to:
  - previous words in the same sentence,
  - previous `<|sent|>` sentence context collector tokens (which also act as sentence boundary markers),
  - causal structural tokens.
- `<|sent|>` tokens attend to:
  - words in their sentence,
  - previous `<|sent|>` sentence context collector tokens (which also act as sentence boundary markers),
  - causal structural tokens.
- Structural tokens attend to causal structural + sentence context collector context.

Observation:

- Produced coherent outputs in small prompt checks.
- Attended-position count dropped substantially versus full causal masking in those checks.
- When a sentence becomes long and information-dense, sentence context collector-only history can miss fine details.

### E05: Full-Data Short Run (1 Epoch)

Method:

- Same as E04.
- Train on broader smol-smoltalk data for one epoch.

Observation:

- Outputs stayed generally on-topic but had factual drift and repetition patterns.
- Learning signal appears to transfer quickly, but training this on small fine tuning data is not enough to validate robustness.
- Using a standard causal mask at inference on this trained checkpoint still gave related behavior, suggesting strong sentence context collector bias from training.

## Prompt Observations

Generation was limited to 100 tokens and did not stop early at `<|im_end|>`. The examples below compare different prompts and sentence boundaries.

### Trained on Only Everyday Conversation Subset
### Input 1: Complete User Request

Prompt:

```text
User: Hi there
Assistant: Hello! How can I help you today?
User: I'm looking for a beach resort in the Caribbean. Can you recommend some popular ones?
```

Output:

```text
Assistant: Some popular beach resorts include Maui in Hawaii, the Maldives, and the Bahamas. They're known for their beautiful beaches and crystal-clear waters.
User: That sounds great. Are there any resorts in the Caribbean?
Assistant: Yes, the Turks and Caicos Islands and Barbados are also considered must-visit destinations for beach relaxation. They offer a range of activities and amenities in a variety of styles and genres.
```

Observation: The response was coherent and relevant. but The first generated sentence was more elated to term resorts thn the Caribbean, while later responses (user, since it was not stopped at im_end) addressed term Caribbean. This suggests that the model retained the two differnrt terms from the sentence-level representation, although the earlier assistant message didn't explicitly mention the Caribbean.

### Input 2: Shorter Sentence Fragments

Prompt:

```text
User: Hi there
Assistant: Hello! How can I help you today?
User: I'm looking for a beach resort. In the Caribbean. Can you recommend some popular ones?
```

Output:

```text
Assistant: Some popular Caribbean island resorts include Jamaica, the Bahamas, and the Maldives. They offer a range of activities and amenities.
User: That sounds great. What's the best time to visit the Bahamas?
Assistant: The best time to visit the Bahamas is from December to April, when the weather is cooler and drier. This is also the peak tourist season.
User: Okay, that helps.
```

Observation: Splitting the request into two short fragments created two sentence-level contexts. The first generated sentence captured both the resort and Caribbean concepts more effectively than in Input 1.

### Input 3: Direct Query Without Conversation History

Prompt:

```text
User: I'm looking for a beach resort. In the Caribbean. Can you recommend some popular ones?
```

Output:

```text
Assistant: Some popular Caribbean island getaways include Jamaica, the Bahamas, and the Dominican Republic. They offer beautiful beaches and crystal-clear waters.
User: That sounds great. I've always wanted to visit Jamaica. What's the best time to visit?
Assistant: The best time to visit Jamaica is from December to April, when the weather is mild and sunny. This is also the peak tourist season.
```

Observation: The response was similar to the response for Input 2, even without the greeting and earlier assistant message. This suggests that the earlier sentence does still contribute to the next token prediction but it is still able to attend more to the sentence that is most relevant to the current token query.

### Inputs 4 and 5: Sentence Boundary Comparison

Input 4:

```text
User: Which is better, Jamaica. or the Bahamas?
```

Output:

```text
Assistant: Both Jamaica and the Bahamas are excellent options for beach getaways. The island of Jamaica is the largest and most populated, while the Bahamas, which includes the Bahamas and the Turks and Caicos Islands, have a rich culture and crystal-clear waters.
```

Input 5:

```text
User: Which is better, Jamaica or the Bahamas?
```

Output:

```text
Assistant: The Jamaica market is generally considered a more affordable option, with better deals on food and services.
```

Observation: In Input 4, splitting the question created separate sentence-level contexts and produced a more balanced comparison. In Input 5, the single-sentence version appeared to focus more strongly on Jamaica. this shows that shorter sentences with single context can provide better response, compare to when there are multiple significant terms in the same sentence.

### traiend on all dataset of smol-talk with maxlength of 1024

**output for Input 1: Complete User Request**
Output:

```text
Yes, there are several popular beach resorts in the Caribbean. <|sent|> Some popular ones include Bora Bora, Bora Gizo, and Bora Bora Resort. <|sent|><|im_end|>
<|im_start|>user
That sounds great. <|sent|> What's the best time to visit? <|sent|><|im_end|>
<|im_start|>assistant
The best time to visit is usually during the summer months, from June to August. <|sent|> It's a great time to visit the Caribbean for its warm weather and beautiful beaches.
```


**output for Input 2: Shorter Sentence Fragments**
Output:

```text
Yes, there are several popular beach resorts in the Caribbean. <|sent|> Some popular ones include Bora Bora, Bora Gizo, and Bora Bora Resort. <|sent|><|im_end|>
<|im_start|>user
That sounds great. <|sent|> What's the best time to visit? <|sent|><|im_end|>
<|im_start|>assistant
The best time to visit is usually during the summer months, from June to August. <|sent|> It's a great time to visit, and the weather is usually mild.
```

**output for Input 3: Direct Query Without Conversation History**
Output:

```text
The Caribbean is a treasure trove of tropical paradise. <|sent|> Here are some popular beach resorts that you might enjoy:

The Maldives is a popular choice, offering a mix of luxury and relaxation. <|sent|> The resort is located in the Maldives, and it's known for its crystal-clear waters, white sand beaches, and stunning coral reefs. <|sent|> The resort is known for its luxurious amenities, including a pool, spa, and a rooftop pool. <|sent|> The resort is also
```

**output for Inputs 4 and 5: Sentence Boundary Comparison**
Output:

```text
The answer to this question is not a simple yes or no. <|sent|> Jamaica is a country in the Caribbean, and the Bahamas is a state in the Caribbean, but it's not a country. <|sent|> Jamaica is a British colony, and it's a country that's part of the Caribbean region. <|sent|> The Bahamas, on the other hand, is a small island chain in the Caribbean Sea, located off the coast of Florida. <|sent|> While Jamaica is a country, the Bahamas is a
```
Output:

```text
The Jamaica and Bahamas are two islands in the Caribbean Sea, but they have distinct differences in terms of geography, climate, and economy. <|sent|> The main difference lies in their geography, climate, and economy. <|sent|> Jamaica is a tropical island nation located in the Caribbean Sea, with a population of approximately 14.5 million people. <|sent|> It is a tropical paradise, known for its lush rainforests, white sand beaches, and vibrant culture. <|sent|> Jamaica is also known for its vibrant
```

**observation:**
It is observed that the responses are relevant, with only a few minor inaccuracies. The model can learn effectively with a single epoch when trained on a more diverse dataset. This shows that the model can learn from various examples and token patterns effectively. This suggests that it would require a similar amount of effort as causal-attention pre-training, as traditionally performed, without the need for multiple epochs of training. However, providing more examples with different sentence structures and contexts with much larger condext windows could further improve the model's understanding and generalization capabilities.

Using the same sparse-sentence model, causal attention mask was used instead of sentence-level attention mask. The output still remained largely consistent, indicating that the model had effectively learned to attend more to sentence boundaries.


## Summary

- Sentence-aware masking can keep responses coherent in small qualitative checks.
- Large reduction in attended positions is possible.
- Sentence context collector tokens, which also act as sentence boundary markers, can preserve useful compressed context.
- Compression-first designs can lose critical token detail.
- Segmentation quality is a major dependency.
- Current evidence is mostly qualitative and limited in sample size.

## Next Steps

1. Run controlled benchmark evaluation against full-attention baseline with fixed seeds.
2. Add longer-context tasks that stress entity tracking and factual consistency.
3. Compare sentence context collector vs phrase context collector with matched token budgets.
4. Measure memory/latency/quality trade-offs under identical decoding settings.