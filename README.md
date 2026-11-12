# GATES — Graph-Augmented Clinical Summarization

**GATES** is a Neuro-Symbolic clinical summarization pipeline that combines Large Language Models (LLMs) with structured Knowledge Graph representations to generate concise, factually grounded summaries from Electronic Health Records (EHRs).

This project was developed as part of the Master's dissertation:

**"Enhancing Clinical Summarization through Knowledge Graph Grounding and Timeline Visualization"**

João Pedro Antão Sobral · Faculty of Engineering, University of Porto (FEUP) · 2026

---

## Project Overview

Electronic Health Records contain large volumes of heterogeneous clinical information, making it difficult for healthcare professionals to identify relevant patient information efficiently. This information overload, often referred to as *Note Bloat*, can negatively affect clinical workflows and increase the risk of overlooking important facts.

GATES explores a neuro-symbolic approach to clinical summarization, combining the language capabilities of LLMs with structured representations of clinical information.

The project was developed and evaluated through three progressive architectural iterations:

- **Iteration 1 — Baseline:** Direct LLM summarization of raw clinical notes, without additional structural constraints.
- **Iteration 2 — NER Grounding:** Clinical Named Entity Recognition (NER) is used to identify relevant entities and guide the generation process.
- **Iteration 3 — Knowledge Graph Grounding:** Extracted clinical entities are consolidated into a structured intermediate representation, reducing redundancy and providing a more reliable factual basis for summary generation.

**Key Results:** The final Knowledge Graph-based approach achieved summary volume reductions of up to 75%, while maintaining strong semantic similarity (BERTScore of approximately 0.90). These results demonstrate the potential of structured clinical representations to improve summary conciseness without substantially compromising clinical meaning.

---

## Architecture

The pipeline follows four main stages:

1. **Clinical Entity Extraction:** Identifies relevant clinical information, including diseases, symptoms, medications, and other patient-related facts.
2. **Knowledge Graph Construction:** Organizes extracted entities into an intermediate structured representation, consolidating redundant or overlapping information.
3. **Clinical Summary Generation:** Uses the structured representation as a factual reference to guide LLM-generated summaries.
4. **Timeline Visualization:** Explores the transformation of structured clinical information into chronological patient timelines.

The architecture aims to combine the flexibility of generative models with the consistency and traceability of structured knowledge representations.

---

## Implementation

The project is organized around three main iterations and a set of evaluation and analysis components.

### Core Pipeline

| Script | Description |
|---|---|
| `it1_baseline.py` | Implements the baseline approach, sending raw clinical notes directly to LLMs for summarization. |
| `it2_ner.py` | Implements NER-based grounding, extracting clinical entities to guide summary generation. |
| `it3_graph.py` | Constructs and filters the intermediate Knowledge Graph, consolidating redundant clinical entities. |
| `it3_summary.py` | Generates clinical summaries using the structured Knowledge Graph as a factual reference. |

### Evaluation and Analysis

| Script | Description |
|---|---|
| `evaluate_metrics.py` | Evaluates generated summaries using operational, lexical, semantic, and factuality metrics. |
| `comparison.py` | Compares results across architectural iterations, calculating performance differences and conducting paired statistical tests. |

### Data and Outputs

- `dataset/` — Contains the input corpus of clinical notes and EHR segments.
- `outputs/` — Stores generated summaries, model responses, execution logs, evaluation results, and cross-iteration comparisons.

---

## Evaluation Framework

GATES uses a multi-dimensional evaluation strategy to assess summary quality, factual consistency, and operational efficiency.

### 1. Operational Efficiency

- **Compression Ratio:** Measures the reduction in summary length relative to the original clinical information.
- **Execution Time:** Measures the time required to process and generate each summary.

### 2. Lexical and Semantic Similarity

- **ROUGE-1, ROUGE-2 and ROUGE-L:** Evaluate lexical overlap between generated and reference texts.
- **BLEU:** Measures n-gram precision.
- **METEOR:** Accounts for lexical variations, including stemming and synonyms.
- **BERTScore (F1):** Uses biomedical language representations to assess semantic similarity beyond exact word matching.

### 3. Factuality and Hallucination Assessment

- **FActScore:** A custom LLM-as-a-Judge evaluation component that decomposes generated summaries into atomic factual claims and verifies them against the original clinical notes.

This evaluation dimension is particularly important in clinical applications, where generated information must remain grounded in the patient's source records.

---

## Supported Language Models

The experiments explore multiple LLM providers and models:

- **OpenAI:** GPT-4o-mini
- **Nebius:** Llama-3.3-70B
- **Google:** Gemini

The multi-provider setup enables comparisons across different language models under the same summarization architecture.

---

## Logging and Reproducibility

Pipeline execution, evaluation procedures, errors, and metric results are recorded using Python's `logging` module.

Execution logs are saved alongside the corresponding experiment outputs, using timestamped log files to support traceability and facilitate comparisons between runs.

---

## Reference Strategy

The evaluation pipeline supports two reference configurations:

| Configuration | Reference | Purpose |
|---|---|---|
| `USE_REFERENCES = False` | Original clinical note | Evaluates generated summaries when gold-standard summaries are unavailable. |
| `USE_REFERENCES = True` | Gold-standard discharge summary | Enables comparison against an existing reference summary. |

The selected reference strategy affects how evaluation metrics should be interpreted, particularly recall-oriented lexical metrics. Semantic similarity and factuality assessments provide complementary perspectives on summary quality.

---

## Technologies

- **Language:** Python
- **Generative AI:** OpenAI, Nebius and Google Gemini APIs
- **Clinical NLP:** Named Entity Recognition (NER)
- **Knowledge Representation:** Structured JSON Knowledge Graph
- **Evaluation:** ROUGE, BLEU, METEOR, BERTScore and FActScore
- **Statistical Analysis:** Paired T-Tests
- **Experiment Management:** Python Logging

---

## Research Context

GATES was developed as part of the Master's dissertation **"Enhancing Clinical Summarization through Knowledge Graph Grounding and Timeline Visualization"** at the Faculty of Engineering of the University of Porto (FEUP).

The research investigates how structured clinical knowledge can support more concise, reliable, and interpretable LLM-generated summaries, addressing challenges such as information overload, factual consistency, and information preservation in Electronic Health Records.

**Author:** João Pedro Antão Sobral  
**Institution:** Faculty of Engineering, University of Porto (FEUP)  
**Year:** 2026
