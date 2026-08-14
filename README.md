# Sentence-Wise Sparse Attention with Sentence-Level KV Cache

> **Research idea:** Reduce long-context Transformer inference cost by treating a **sentence as the primary unit of long-range information**, while retaining full token-level information only for the **currently active/last sentence**.

## 1. Motivation

Standard Transformer attention and KV caching operate at the **token level**. For a long document, this means that every previously generated token may remain in the KV cache even when its fine-grained word-level information is no longer necessary.

The proposed approach introduces a hierarchy:

```text
Document
│
├── Sentence 1 ──► [SENT₁] ──► compact representation
├── Sentence 2 ──► [SENT₂] ──► compact representation
├── Sentence 3 ──► [SENT₃] ──► compact representation
│
└── Current Sentence
      ├── word₁
      ├── word₂
      ├── word₃
      └── ...
```

The key idea is:

**Words provide detailed local information; sentence tokens provide compressed long-range information.**

---

## 2. Core Architecture

Each sentence begins with a special **sentence token**, similar in spirit to a `[CLS]` token.

For example:

```text
Input:

[S₁] The cat sat on the mat .
[S₂] It was sleeping peacefully .
[S₃] Suddenly a dog entered the room .

        ↓

Sentence representation:

[S₁]  [S₂]  [S₃]
 │     │     │
 ▼     ▼     ▼
 KV    KV    KV

Current sentence:
[S₃] Suddenly a dog entered the room .
      └──── word-level tokens ────┘
```

The model performs **local token attention inside a sentence**, rather than allowing every token to attend densely to the entire document.

### Local attention

```text
             Sentence Sᵢ
        ┌───────────────────┐
        │ [Sᵢ] w₁ w₂ w₃ w₄ │
        └───────────────────┘
          ↕   ↕  ↕  ↕  ↕
        Local token attention
```

Tokens in the same sentence can exchange detailed information.

The sentence token acts as the **information bottleneck** between sentences.

---

## 3. Cross-Sentence Information Flow

After local attention, the sentence token aggregates information from its sentence.

```text
        Sentence i
 ┌─────────────────────────┐
 │ [Sᵢ]  w₁  w₂  w₃  w₄   │
 │   ↕    ↕   ↕   ↕   ↕    │
 │       Local Attention   │
 └────────────┬────────────┘
              │
              ▼
           [Sᵢ]
       sentence summary
              │
              ▼
      ┌─────────────────┐
      │ Other sentences │
      │ S₁ S₂ ... Sᵢ₋₁ │
      └─────────────────┘
```

Thus, long-range communication happens primarily through:

```text
word tokens
     │
     ▼
local sentence attention
     │
     ▼
sentence token
     │
     ▼
cross-sentence attention
     │
     ▼
sentence token
```

This avoids repeatedly exchanging all historical word tokens across the entire context.

---

## 4. Head-Level Sparse Attention

A possible implementation is to divide attention heads into two functional groups.

### Local heads

Local heads operate on word-level tokens inside the current/local sentence.

```text
Local Head

[Sᵢ] ─ w₁ ─ w₂ ─ w₃ ─ w₄
 │     ↕    ↕    ↕    ↕
 └──────── local ────────┘
```

They capture:

* syntax
* word relationships
* local semantic dependencies
* short-range positional information

### Sentence/global heads

A smaller set of heads operates primarily on sentence tokens.

```text
Sentence Head

[S₁] ───── [S₂] ───── [S₃] ───── [S₄]
  ╲          │           │          ╱
   ╲─────────┴───────────┴─────────╱
```

These heads capture:

* document-level dependencies
* topic continuity
* information from previous sentences
* long-range semantic relationships

This creates a **two-level attention hierarchy**:

```text
                 Transformer Layer
                        │
              ┌─────────┴─────────┐
              │                   │
        Local Attention     Sentence Attention
              │                   │
       word-level detail     sentence-level
                              information
              │                   │
              └─────────┬─────────┘
                        ▼
                 Next Transformer
```

---

## 5. Token Dropping / Compression

The main efficiency gain occurs after a sentence is completed.

Instead of keeping all its word tokens for future decoding:

```text
Before:

[S₁] w₁ w₂ w₃ w₄ w₅
[S₂] w₁ w₂ w₃ w₄
[S₃] w₁ w₂ w₃ w₄ w₅
[S₄] w₁ w₂
```

the cache becomes approximately:

```text
After sentence completion:

[S₁]   [S₂]   [S₃]   [S₄]
 │      │      │      │
 KV     KV     KV     KV

        +
        
Current sentence:
[S₄] w₁ w₂
```

In other words:

> **Historical sentences → sentence-level KV only**
> **Current/last sentence → sentence token + word-level KV**

The word-level KV pairs of completed sentences can therefore be removed or compressed.

---

## 6. Last-Sentence Token Retention

During autoregressive generation, the model maintains detailed token-level information only for the currently active sentence.

For example:

```text
Document context

S₁: The company launched a new product.
     ↓
   [S₁]                  ← keep sentence KV

S₂: The product became popular very quickly.
     ↓
   [S₂]                  ← keep sentence KV

S₃: Customers especially liked its battery
     ↓
   [S₃] + Customers + especially + liked + its + battery
         └──────── word-level KV ────────┘
```

When the model generates:

```text
"life."
```

and the sentence ends, the word tokens of S₃ can be compressed into `[S₃]`.

Then the next sentence starts:

```text
[S₁] [S₂] [S₃] [S₄]
                  │
                  └── current sentence word tokens
```

This creates a moving **high-resolution window at the sentence level** rather than a conventional token-level sliding window.

---

## 7. KV Cache Design

A conventional decoder stores:

```text
KV cache:

S₁: token KV token KV token KV token KV
S₂: token KV token KV token KV
S₃: token KV token KV token KV
S₄: token KV token KV
```

The proposed cache instead stores:

```text
Sentence-level cache:

S₁: [SENT] KV
S₂: [SENT] KV
S₃: [SENT] KV
S₄: [SENT] KV

Current sentence:

[S₄] KV
w₁ KV
w₂ KV
w₃ KV
...
```

Conceptually, if a document contains `N` sentences with an average of `T` tokens per sentence, the historical KV storage changes from approximately:

```text
O(N × T)
```

token representations to approximately:

```text
O(N)
```

sentence representations, plus the tokens of the active sentence.

This does not automatically guarantee an `N×T` speedup—the attention computation, projection, memory bandwidth, and implementation details still matter—but it provides the architectural opportunity for a substantial reduction in KV-cache size.

---

## 8. End-to-End Data Flow

```text
                 INPUT DOCUMENT
                       │
                       ▼
              Sentence segmentation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        S₁ tokens     S₂ tokens    S₃ tokens
          │            │            │
          ▼            ▼            ▼
      Local Attn    Local Attn    Local Attn
          │            │            │
          ▼            ▼            ▼
        [S₁]          [S₂]          [S₃]
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
              Sentence-level cache
                       │
                       ▼
              Next-token prediction
                       │
                       ▼
          Keep only current sentence
             at token resolution
```

---

## 9. Proposed Attention Pattern

A simplified attention matrix could look like:

```text
             KEY TOKENS
        S₁     S₂     S₃     Current
      ┌──────┬──────┬──────┬──────────┐
S₁    │  ●   │      │      │          │
      ├──────┼──────┼──────┼──────────┤
S₂    │      │  ●   │      │          │
      ├──────┼──────┼──────┼──────────┤
S₃    │      │      │  ●   │          │
      ├──────┼──────┼──────┼──────────┤
Current│     │      │  ●   │ ● ● ● ●  │
      └──────┴──────┴──────┴──────────┘

● = permitted sparse attention
```

At the token level, the current sentence receives detailed attention:

```text
Current sentence:

[S₄] w₁ w₂ w₃ w₄ w₅

       ↕  ↕  ↕  ↕  ↕
     dense/local attention
```

while historical sentences are represented through their sentence tokens:

```text
[S₁] ←→ [S₂] ←→ [S₃] ←→ [S₄]
```

---

## 10. Potential Training Objective

The model can initially be trained using the standard next-token prediction objective:

[
\mathcal{L}_{LM}
================

-\sum_t \log P(x_t \mid x_{<t})
]

The important architectural constraint is that historical word-level representations are not directly available after sentence compression.

A useful additional objective could encourage the sentence token to preserve information needed for future prediction:

[
\mathcal{L}
===========

\mathcal{L}*{LM}
+
\lambda \mathcal{L}*{sentence}
]

where `L_sentence` could be a sentence-reconstruction, contrastive, distillation, or future-token prediction objective.

For example:

```text
Sentence tokens
      │
      ▼
Predict information required by
future sentences
      │
      ▼
Sentence representation becomes
useful for long-range prediction
```

This is particularly important because simply replacing many token KV pairs with a single `[SENT]` vector creates a strong information bottleneck.

---

## 11. Why This Could Be Interesting

The central research hypothesis is:

> **For long-context autoregressive generation, once a sentence is completed, much of its word-level information may be unnecessary for future prediction if an appropriate sentence-level representation is retained.**

This gives a natural hierarchy:

```text
                  INFORMATION
                      │
          ┌───────────┴───────────┐
          │                       │
     Short-range              Long-range
          │                       │
          ▼                       ▼
    Word-level KV           Sentence-level KV
          │                       │
      High detail             Low detail
      High memory             Low memory
```

The interesting research question is therefore not simply *"Can we prune tokens?"*, but:

**"Can sentence boundaries provide a principled compression boundary for autoregressive KV caches?"**

---

## 12. Main Experiments to Validate the Idea

A strong initial experiment could compare:

| Method                   | Historical context  | Current sentence | KV cache                      |
| ------------------------ | ------------------- | ---------------- | ----------------------------- |
| Dense Transformer        | All tokens          | All tokens       | Full                          |
| Token pruning            | Selected tokens     | Tokens           | Reduced                       |
| Sliding-window attention | Recent tokens       | Tokens           | Window                        |
| **Proposed method**      | **Sentence tokens** | **All tokens**   | **Sentence + current tokens** |

Measure:

* perplexity
* long-context retrieval accuracy
* Needle-in-a-Haystack
* LongBench
* generation quality
* KV-cache memory
* prefill latency
* decoding latency
* tokens/sec
* maximum context length
* quality vs. cache-compression ratio

A particularly important ablation would be:

```text
Full tokens
     ↓
Last 2 sentences
     ↓
Last 1 sentence
     ↓
Last sentence + sentence tokens
     ↓
Sentence tokens only
```

This would directly test how much information is lost when moving from token-level to sentence-level memory.

---

# Related Work

### 1. Hierarchical Document Transformer (HDT)

[HDT — Hierarchical Document Transformer](https://arxiv.org/abs/2407.08330?utm_source=chatgpt.com)

HDT is probably the **closest architectural precedent**. It introduces sentence, section, and document anchor tokens and uses hierarchical sparse attention so tokens communicate with siblings and parent/child anchors.

**Difference:** HDT is primarily a hierarchical sparse-attention architecture for processing structured documents, whereas the proposed idea specifically targets **autoregressive inference and KV-cache compression**, with completed sentences collapsing to sentence-level KV while the current sentence retains word-level KV.

### 2. SentenceKV

[SentenceKV — Efficient LLM Inference via Sentence-Level Semantic KV Caching](https://arxiv.org/abs/2504.00970?utm_source=chatgpt.com)

SentenceKV is another very close recent direction: it groups tokens into sentences, creates compact sentence-level semantic vectors, and selectively retrieves token KV pairs during decoding.

**Difference:** SentenceKV primarily uses sentence representations to **select/retrieve important token KV pairs**, whereas this proposal is more aggressive: **completed sentence word tokens can be discarded from the active cache and represented directly by a sentence token**, while only the current/last sentence retains its individual word tokens.

### 3. Longformer

[Longformer documentation](https://huggingface.co/transformers/v4.8.2/model_doc/longformer.html?utm_source=chatgpt.com)

Longformer combines local attention with a small number of globally attending tokens, reducing the quadratic attention pattern to a local-window pattern.

**Difference:** Longformer is based primarily on fixed local/global attention patterns; the proposed method introduces **sentence boundaries as the compression hierarchy** and dynamically transitions completed sentences from token-level to sentence-level memory.

### 4. Token Sparse Attention

[Token Sparse Attention — Official implementation](https://github.com/dongwonjo/Token-Sparse-Attention?utm_source=chatgpt.com)

Token Sparse Attention dynamically selects tokens on a per-head basis during long-context inference and allows token relevance to be reevaluated across layers and heads.

**Difference:** It remains fundamentally **token-level sparsification**. The proposed approach instead exploits the semantic structure of language—**sentences—as the unit at which historical information is compressed**.

---

## One-Sentence Summary

**Sentence-Wise Sparse Attention compresses completed sentences into persistent sentence-level KV representations, while retaining full word-level KV only for the current sentence, creating a hierarchical long-context Transformer that trades fine-grained historical information for much smaller and semantically structured memory.**

> **Potential novelty:** The strongest research angle is not simply "sentence-level attention"—which has prior work—but the combination of **sentence-boundary-triggered token eviction + persistent sentence-token KV cache + current-sentence full-resolution KV + sparse/head-specific information flow** for autoregressive decoding.
