## Purpose
- Reduce KV memory and full attention while keeping generation quality close to full attention.

## Experiments

### E01 - Sentence Pool 
- use the full attention pretrained model and pool the tokens form the sentence into a single representation.
- Result:
full replacement by pooled vectors loses important token detail, even with keeping the last few sentences uncompressed.

### E02 - Sentence level Representation in Key cache
- pool the kv cache of the tokens from the same sentence and use that for the retrieval top relevant sentence and get all the tokens from that sentence.
- Result:
- in here the a simple pooling of key of all tokens of the sentence was not able to provide relative representation so that query could get high score for those sentence-level representations. due to this it was not able to get the top relevant sentences accurately.


### E03 - Sentence level Representation with local and global attention (Two-Pass)
- intriduce new special token `<|sent|>` which would be the cls token for sentence-level representation.
- The model will use local attention for word tokens within a sentence and global attention for sentence-level tokens.
- every penultimate lyer will have two pass with same set of qkv weight for locla and global attention.
- last layer will have only the local attention and the next token will be predicted by only the word token.
result:
- model was not able to converge. two pass intriudced complexity and overhead for the head token to attend to other sent tokens, as it can only flow through the sentence-level representations.

### E04 - Sentence-aware with Single-Pass where word token attends to other sentence representation tokens directly
- introduce the special token `<|sent|>` for sentence-level representation.
which also acts as the sentence marker, indicating the end of a sentence and providing a compact representation of that sentence for next word tokens.
- placed sentence markers at the end of sentences so they can summarize the words that precede them.
- used spaCy sentence segmentation to insert `<|sent|>` after each sentence.
- used one attention pass per decoder layer; no separate local/global attention passes were used.
- applied the same sentence-aware attention mask during training and inference. such that,
  - word tokens attends to any previous words otken in the same sentence and to the previous sentence representation tokens (`<|sent|>`) and structural tokens.
  - sentence-level tokens (`<|sent|>`) attend to all word tokens in the same sentence and to all previous sentence-level tokens and structural tokens.
  - structural tokens attend to only other structural tokens and sentence-level tokens.
  
- Result:
  - the model generated coherent and mostly relevant responses in the evaluation prompts.
  - the sentence-aware mask reduced the number of attended positions by approximately 70% compared with full causal attention for these examples.
  - the sentence marker can act as a compact representation of the sentence it concludes, allowing later tokens to attend to the marker instead of every word in earlier sentences.
  - it failed to capture the full context of the sentence when it is large, leading to potential loss of important information for subsequent word tokens.


## Baseline Snapshot
- Retrieval/compressed(pooled) variants currently trade memory for notable quality degradation.
- The single-pass sentence-aware model has produced the qualitative results documented below, a larger controlled quality comparison is still needed.




## Attention Design

The learned `<|sent|>` token marks the end of each sentence. spaCy sentence segmentation was used to insert the token at sentence boundaries. The model uses a single attention pass per decoder layer; no separate local and global passes are used.

The attention mask preserves causal attention while limiting each token to relevant context:

- A word token attends to previous word tokens in the same sentence and to the sentence markers (`<|sent|>`) of previous sentences. It also attends to the relevant structural tokens, such as `<|im_start|>`, `<|im_end|>`, `system`, `user`, and `assistant`.
- A sentence token (`<|sent|>`) attends to the previous word tokens in its sentence, the sentence markers of previous sentences, and the structural tokens.
- A structural token attends only to structural tokens and sentence markers allowed by the causal mask.

The intuition is that the sentence-end marker summarizes the sentence it concludes. Later tokens can therefore use this compact representation instead of attending to every word in earlier sentences.

The model was trained with this sentence-level attention mechanism using `HuggingFaceTB/SmolLM2-135M-Instruct` and the `everyday-conversations` subset of `HuggingFaceTB/smol-smoltalk` for 10 epochs. The results provide a proof of concept for reducing attention computation and dependence on a full token-level KV cache while preserving useful context.

## Evaluation Results

Generation was limited to 100 tokens and did not stop early at `<|im_end|>`. The examples below compare different prompts and sentence boundaries.

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

## Key Takeaways

These examples suggest that `<|sent|>` can capture sentence-level information and provide it to later tokens without requiring them to attend to every word in the earlier context. Shorter sentence fragments sometimes produced more focused retrieval of the relevant context than longer sentences. Across these examples, the number of attended positions was approximately more than 60% lower than with full causal attention.

The results are promising but preliminary. The prompts were few, the training subset was small, and sentence segmentation directly affects the available sentence-level contexts. Larger-scale evaluation is needed before drawing conclusions about response quality or generalization.

exp:e05 The same above experiemtn was done with smol-talk full data with maxlenght of 1024 instead of everyday conversations for one epcoh, below is the result for same input.

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

## Next Steps

The `<|sent|>` token was also a token similar to any other token in the vocabulary, but it was able to capture the sentence-level context. This was due to the training with custom masking strategies that emphasized sentence boundaries.
so could this means any other tokens could also serve as context-capturing tokens if trained appropriately?

The next experiment will around this idea to test whether existing tokens can serve as context-capturing tokens instead of introducing a dedicated `<|sent|>` token. Can a token at the end of the phrase could summarize the preceding phrase, reducing the number of tokens that later positions need to attend to without compressing an entire sentence into one representation.

For example, in the sentence "The Jamaica market is generally considered a more affordable option, with better deals on food and services," candidate phrase-ending tokens could be:

- `market` for "The Jamaica market"
- `considered` for "is generally considered"
- `option` for "a more affordable option"
- `deals` for "better deals"
- `food` for "on food"
- `services` for "and services"

If each word were represented by one token, this example would contain approximately 18 tokens and could be represented by six context-capturing tokens and this is a significant reduction in the number of tokens that later positions need to attend to.

The context-capturing token should appear at the end of the phrase it represents, and the attention mask should allow later tokens to attend primarily to these tokens rather than to every token in the phrase.

The main challenge is identifying phrase boundaries reliably. Future experiments should evaluate whether phrase-ending tokens preserve enough information when a phrase combines subjects, objects, adjectives, or adverbs. The results should be compared using both response quality and the number of attended positions.