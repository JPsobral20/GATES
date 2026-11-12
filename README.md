# Clinical Summarization - Neuro-Symbolic Knowledge Graph Pipeline

> Part of the dissertation: **"Enhancing Clinical Summarization through Knowledge Graph Grounding and Timeline Visualization"**
> João Pedro Antão Sobral - FEUP, 2026

---

## Project Overview

This project implements and evaluates a **Neuro-Symbolic clinical summarization pipeline**. It aims to solve the severe information overload in Electronic Health Records (EHRs), known as *"Note Bloat"*, and eliminate the risk of LLM hallucinations. 

The architecture was developed and tested across **three progressive iterations** using state-of-the-art Large Language Models (OpenAI GPT-4o-mini, Nebius Llama-3.3-70B, and Google Gemini):
* **Iteration 1 (Baseline):** Raw LLM generation (Prone to hallucination and omission of vital facts).
* **Iteration 2 (NER Grounding):** Entity extraction anchoring (Improved accuracy but caused severe *Note Bloat* / +55% compression ratio).
* **Iteration 3 (Knowledge Graph):** The final solution. Filters and structures entities into a logical JSON graph before generation. 

**Key Achievement:** Iteration 3 successfully reduced summary volume by up to **-75%**, effectively curing *Note Bloat*, while maintaining clinical semantic accuracy intact (**BERTScore ~0.90**) and preventing hallucinations.

---

## Pipeline Architecture

1. **Entity Extraction (NER):** Clinical facts (diseases, symptoms, medications) are extracted from raw EHRs.
2. **Graph Construction & Filtering:** Entities are consolidated into an intermediate structured Knowledge Graph (JSON) to remove redundancies.
3. **Summary Generation:** The LLM consumes the structured Graph as a strict factual anchor to write a concise clinical narrative.
4. **Visualization (WIP):** Transforming the JSON Graph into interactive, chronological patient timelines.

---

## File Structure & Script Descriptions

### Core Pipeline Scripts
* **`it1_baseline.py`**: Establishes the project control group. It sends raw, un-structured clinical notes directly to the LLMs (OpenAI, Nebius, Google) for summarization without any symbolic or structural restrictions.
* **`it2_ner.py`**: Implements the second iteration by extracting clinical entities via Named Entity Recognition (NER) and forcing the LLM to anchor its narrative around them. This improves factual coverage but leads to severe word redundancy (*Note Bloat*).
* **`it3_graph.py`**: The structural core of the final Neuro-Simbólica architecture. It ingests extracted entities, deduplicates lexically redundant concepts, and organizes them into a clean, hierarchical intermediate **Knowledge Graph (JSON)**.
* **`it3_summary.py`**: The narrative generation script for the final iteration. It feeds the cleaned JSON Graph into the LLM as a rigid factual constraint, forcing the model to generate a highly compressed, noise-free clinical summary.

### Evaluation & Analysis Scripts
* **`evaluate_metrics.py`**: The complete evaluation suite. Automatically processes a specific run folder to compute operational metrics (Compression Ratio, Execution Time), lexical overlap (ROUGE-1/2/L, BLEU, METEOR), semantic alignment (BiomedBERT BERTScore), and hosts the custom automated **FActScore** (LLM-as-a-Judge) factual auditor.
* **`comparison.py`**: The statistical engine of the project. It compares metrics across different iterations (e.g., IT1 vs IT2, or IT2 vs IT3) to generate delta analysis, percentage improvements, and computes **Paired T-Tests** to validate statistical significance.

### Data & Directories
* **`dataset/dataset/`**: Data layer containing the input corpus of raw, unstructured clinical notes and EHR segments (`.txt` files).
* **`outputs/`**: Generated artifacts organized by run name and timestamp, including detailed model responses (`openai/`, `nebius/`, `google/`), execution logs, and a `comparisons/` directory containing cross-iteration significance tables.

---

## Evaluation Strategy

The pipeline goes beyond traditional (and often flawed) N-gram matching by employing a comprehensive suite of metrics:

### 1. Operational Efficiency
* **Compression Ratio:** Measures summary conciseness (Crucial for proving *Note Bloat* reduction).
* **Execution Time (sec):** System latency and computational efficiency.

### 2. Lexical & Semantic Quality
* **ROUGE (1, 2, L) & BLEU:** Measures exact-word overlap. Used to demonstrate the limitations of classical metrics when summaries are highly compressed.
* **METEOR:** A more flexible lexical metric that accounts for stemming and synonymy, providing a better baseline for natural language variation than exact-match ROUGE.
* **BERTScore (F1):** Leverages `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` to prove that clinical meaning is preserved even when the text length is drastically reduced.

### 3. Factuality (Hallucination Mitigation)
* **Custom FActScore (LLM-as-a-Judge):** Dynamically breaks the generated summary into atomic facts and verifies them against the original clinical note using a rigid JSON-mode LLM evaluator to ensure 100% factual grounding.

---

## Logging

All pipeline activity — generation, evaluation, errors, and metric results — is captured by Python's `logging` module and written simultaneously to the console and to `execution_{TIMESTAMP}.log` in the run's respective output folder.

---

## Notes on Reference Strategy

| Mode | Reference used | Best for |
|---|---|---|
| `USE_REFERENCES = False` | Original clinical note | When no gold summaries exist (Current Setup) |
| `USE_REFERENCES = True` | Gold discharge summary | Ideal evaluation setup |

The current setup (`False`) is a pragmatic workaround. Scores should be interpreted with this in mind, particularly Recall-based metrics like ROUGE. However, semantic metrics (BERTScore) and factuality metrics (FActScore) are exceptionally robust in this setup as they verify meaning and facts directly against the source text.