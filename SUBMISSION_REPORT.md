# Intent-Aware Training Objectives for Reducing Surface-Form Bias in Next-Token Prediction

**DA624 — Natural Language Processing with Large Language Models, IIT Guwahati**  
**Team:** Harsh Vardhan Singh Bhati, Pranav Kumar Rachit, Sahastransh Mandhana, Shivasheesh Dubey

## Abstract

Standard next-token prediction (NTP) treats the observed continuation as the only correct target even when several contextually interchangeable continuations are semantically acceptable. We implement a reduced-scale concept-aware training pipeline inspired by Zhang, Jurafsky, and Shani (2026). The pipeline identifies single-token nouns, verbs, and adjectives; obtains high-probability candidate continuations from a base language model; filters candidates with an instruction-tuned model; and fine-tunes Qwen3-1.7B-Base with QLoRA under a loss that rewards probability mass assigned to the resulting contextual concept set. We compare standard NTP, the thresholded concept objective, and an extension that additionally preserves the base model's relative preference among concept members through a KL regularizer. Evaluation includes global perplexity, content-word perplexity and accuracy, out-of-domain OpenWebText evaluation, semantic-similarity Spearman correlation, and paired-bootstrap confidence intervals.

## 1. Motivation

A single prefix can admit multiple natural continuations. Under standard cross-entropy, probability assigned to a semantically valid alternative is nevertheless treated as probability assigned to an incorrect class. This is a form of surface-form competition. Our central question is whether supervision over a set of contextually interchangeable tokens can improve semantic behavior without unacceptable degradation in ordinary language modeling.

## 2. Method

### 2.1 Content-word extraction

We stream C4 and retain text sequences of moderate length. spaCy identifies NOUN, VERB, and ADJ tokens. A target is kept only when its character span aligns to exactly one Qwen tokenizer token in context. Hyphen/apostrophe fragments and one-character artifacts are rejected. Training/evaluation sequences are truncated to 256 LM tokens.

### 2.2 Contextual concept-set construction

For each retained target at token position t, Qwen3-1.7B-Base produces a next-token distribution from logits at t-1. We retain high-probability whole-word, single-token candidates, then ask Qwen3-1.7B to select only candidates that can replace the target in the exact original context without materially changing meaning or grammatical role. The observed target is always included in the final concept set.

### 2.3 Training objectives

Let S_t be the concept set for a target position and p_theta(w|x_<t) the model distribution. The concept-set mass is

P_theta(S_t|x_<t) = sum_{w in S_t} p_theta(w|x_<t).

The thresholded concept loss is -log P_theta(S_t|x_<t) when the mass is below 0.6 and zero otherwise. The reproduced combined objective is

L = (1-lambda) L_NTP + lambda L_concept.

We additionally implement a context-preservation regularizer. During concept-set construction we store the base model's normalized distribution q over the surviving set. During fine-tuning, the current model distribution is renormalized within the same set to obtain r. The extension adds beta * KL(q || r), encouraging concept mass to increase without erasing the base model's context-sensitive ranking among alternatives. beta=0 recovers the reproduced objective.

### 2.4 Parameter-efficient fine-tuning

The base model is Qwen3-1.7B-Base. Training uses 4-bit NF4 quantization and LoRA on q/k/v/o attention projections. The default learning rate is 7e-5, with cosine scheduling, FP16 computation, gradient checkpointing, and a reduced batch size suitable for a Tesla T4. The emergency configuration uses 120 C4 sequences, at most four concept targets per sequence, two epochs, and seed 42. A paper-scale configuration file is also supplied for 2,000 sequences and a broader lambda sweep.

## 3. Sanity Check

Before corpus-scale processing we tested the prefix **"This website is safe to"**. The observed target `browse` received probability 0.274158, while the manually specified concept set received total probability mass 0.649606. The exact-token NLL was 1.2941; raw concept NLL was 0.4314. Because concept mass exceeded the 0.6 threshold, the thresholded concept loss was zero. This verifies the intended loss behavior but is not treated as a benchmark result.

## 4. Results

### 4.1 In-domain C4

| Model | Global PPL | Content PPL | Content Accuracy | Mean Concept Mass |
|---|---:|---:|---:|---:|
| Base | — | — | — | — |
| Ntp | — | — | — | — |
| Concept | — | — | — | — |
| Ours | — | — | — | — |

### 4.2 Out-of-domain OpenWebText

| Model | Global PPL | Content PPL | Content Accuracy | Mean Concept Mass |
|---|---:|---:|---:|---:|
| Base | — | — | — | — |
| Concept | — | — | — | — |
| Ours | — | — | — | — |

### 4.3 Human semantic-similarity benchmarks

Spearman rank correlation is computed from cosine similarity of mean-pooled final-layer representations. The emergency run caps each benchmark at 300 pairs; remove `--max-pairs` for full-benchmark evaluation.

| Model | MEN | WordSim353 | SimLex999 | STS-B |
|---|---:|---:|---:|---:|
| Base | — | — | — | — |
| Concept | — | — | — | — |
| Ours | — | — | — | — |

### 4.4 Statistical validation

Not generated yet.

## 5. Discussion

The central comparison is Base vs NTP vs Concept. If concept training lowers content-word perplexity / raises content accuracy or semantic Spearman while global perplexity remains close to the NTP baseline, that supports the hypothesis that set-level supervision reduces surface-form competition. The context-preserving model is an exploratory extension: improvement over the unregularized concept model would suggest that increasing set mass and retaining contextual preference can be complementary; failure to improve is still informative because it tests whether the extra constraint is unnecessarily restrictive.

The emergency configuration is intentionally small and therefore high-variance. Results should be interpreted as reduced-scale evidence, not as a replication of the original paper's absolute numbers. The supplied full configuration increases the sample count, candidate pool, supervision density, epoch count, lambda sweep, and seeds.

## 6. Limitations

Concept sets inherit errors from POS tagging, tokenizer alignment, the base model's candidate pool, and the instruction-model filter. Only single-token lexical alternatives are modeled, so multi-token paraphrases are excluded. The reduced T4-scale experiment is substantially smaller than large-scale language-model pretraining. Semantic benchmarks measure representation alignment, not general human alignment. Finally, the context-preservation term is an engineering extension evaluated within this course project and should not be described as globally novel without a dedicated literature search.

## 7. Reproducibility

All stages are scripted: preprocessing, top-k generation, contextual filtering, deterministic splitting, QLoRA fine-tuning, in-domain/OOD evaluation, semantic benchmarks, and paired bootstrap intervals. `notebooks/01_submission_pipeline.ipynb` provides a Colab-oriented execution path and `scripts/run_fast_pipeline.sh` runs the same emergency configuration from the command line.

## References

1. Christine Zhang, Dan Jurafsky, Chen Shani. *Learning Concepts, Not Tokens: Self-Supervised Semantic Alignment for Language Models.* arXiv:2603.29123, 2026.
2. Ari Holtzman et al. *Surface Form Competition: Why the Highest Probability Answer Isn't Always Right.* EMNLP, 2021.
3. Colin Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR, 2020. (C4)
4. Felix Hill, Roi Reichart, Anna Korhonen. *SimLex-999: Evaluating Semantic Models with Genuine Similarity Estimation.* Computational Linguistics, 2015.
5. Elia Bruni, Nam-Khanh Tran, Marco Baroni. *Multimodal Distributional Semantics.* JAIR, 2014. (MEN)
6. Lev Finkelstein et al. *Placing Search in Context: The Concept Revisited.* WWW, 2002. (WordSim-353)
7. Daniel Cer et al. *SemEval-2017 Task 1: Semantic Textual Similarity.* SemEval, 2017. (STS-B)
