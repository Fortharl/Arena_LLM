# Arena LLM — EMR Summarization Pipeline for CT Abdomen

**Hackathon:** Арена LLM — «Разработка методов генерации и автоматизированной оценки качества суммаризации медицинских текстов»  
**Team:** КиберДоктор Кто, Сеченовский Университет  
**Members:** Немченко И.А., Калмыкова В.И., Абдулова А.Р., Кириллов Е.О., Куправа Т.Ш.

---

## Overview

This repository contains a fully local, privacy-preserving pipeline for automatic summarization of Electronic Medical Records (EMR) prior to CT abdominal examination, together with an LLM-as-judge evaluation system.

The pipeline is designed around three constraints specific to clinical AI applications: legal restrictions on transmitting medical data to external APIs (Federal Law No. 152-FZ), the need for reproducible and hallucination-resistant structured output, and the absence of reference summaries against which classical similarity metrics could be applied.

---

## Repository Structure

```
.
├── graph_summary_V6.py       # Summarization pipeline (LangGraph + Outlines)
├── evaluation_V5.py          # LLM-as-judge evaluation pipeline (LangGraph + Outlines)
├── schemas_summary.py        # Pydantic schemas and prompts for the summarization graph
├── schemas_evaluation.py     # Pydantic schemas and prompts for the evaluation graph
└── КиберДоктор_Кто_Арена_LLM.pdf   # Presentation slides
```

---

## Architecture

### Summarization Pipeline (`graph_summary_V6.py`)

The pipeline follows a two-stage decomposition: structured fact extraction followed by text synthesis. The key insight is that fact extraction and narrative synthesis are cognitively distinct tasks — attempting both in a single prompt degrades performance on small models.

**Stage 1 — Structured extraction (4 sequential LangGraph nodes):**

| Node | Class | Extracted data |
|---|---|---|
| Node 1 | `SafetyExtractor` | Contrast allergy, renal function (creatinine, eGFR), anticoagulants, metformin, nephrotoxic drugs, diuretics |
| Node 2 | `AnatomyClinicalExtractor` | Chief complaints, disease timeline, GI diagnoses, abdominal surgeries, smoking history |
| Node 3 | `OncoExtractor` | All oncological diagnoses (site, TNM stage, treatment, response, remission) |
| Node 4 | `LabImgExtractor` | Laboratory values with units and dates; imaging findings with modality and organ |

Each extraction node uses constrained decoding via **Outlines** against a Pydantic-defined JSON Schema. This eliminates at the generation level — not post-processing — the class of errors typical of free-form generation: wrong field types, missing required fields, numeric values returned as strings.

**Stage 2 — Summary composition (Node 5 `SummaryComposer`):**

The composer receives as input not the raw EMR text but the aggregated JSON objects produced by the four extraction nodes. The model operates over already-extracted, validated facts rather than unstructured text of up to 25,000 characters.

Patient demographics (age, sex) are extracted separately via regex before the graph is invoked.

### Evaluation Pipeline (`evaluation_V5.py`)

The evaluation system applies the same architecture — local model, Outlines, LangGraph — to the task of quality assessment. Input is a (source EMR, summary) pair; output is a structured `EvalReport` object.

**Scoring system: 5 criteria, 20 binary checks, maximum score 20.**

| Criterion | Checks | Max |
|---|---|---|
| Safety (CRITICAL ALERTS) | allergy, creatinine, eGFR, nephro diagnoses, nephro drugs | 5 |
| Completeness | complaints, anamnesis, oncology, lab/imaging | 4 |
| Factual accuracy | numbers, diagnoses, hallucinations, interpretation | 4 |
| Structure and format | sections, conciseness, volume | 3 |
| Clinical relevance | CT priority, onco-nephro context, noise, actionable items | 4 |

**Zero-Tolerance principle:** a check receives 1 only if the information is present, numerically exact, and not generalized. Four violation types are penalized: OMISSION, GENERALIZATION, DISTORTION, and UNCERTAINTY (writing "not specified" when the source contains the data).

Each criterion produces a `rationale` field with a concrete description of violations. The top-level `summary_notes` field summarizes the most critical omissions and distortions.

**Why not ROUGE/BLEU/BERTScore:** these metrics measure surface similarity to a reference text and cannot detect loss of clinically significant information or factual distortions. Additionally, no reference summaries were provided in the hackathon dataset, making reference-based metrics inapplicable.

---

## Technology Stack

| Component | Role |
|---|---|
| `outlines` | Constrained decoding at the logit level via JSON Schema finite automaton. Guarantees structurally valid output at the generation stage. |
| `pydantic` (BaseModel + Field) | Declarative schema definition. `Field(description=...)` serves simultaneously as JSON Schema source for Outlines and as per-field instruction to the model. |
| `langgraph` | Node orchestration as a directed acyclic graph with typed shared state. Provides the foundation for future agentic (feedback-loop) architecture. |
| `llama_cpp` | Local model inference without external API calls. |
| **LLM** | `ai-sage/GigaChat3.1-10B-A1.8B` (GGUF q6_K quantization, MIT License). MoE architecture. |

**Hardware:** MacBook Pro M1 Pro, 16 GB RAM. No GPU cluster required. Generation throughput: up to 60 tokens/sec.

---

## Results

Evaluated on 6 source EMR texts. The pipeline ranked first among 7 competing solutions with a mean score of **9.8 / 20**, outperforming models up to 70B parameters.

Mean generation time per summary: ~340 seconds.  
Mean evaluation time per (source, summary) pair: ~190 seconds.

---

## Design Rationale

**Local inference over commercial APIs:** medical EMR data is classified as a special category of personal data under Federal Law No. 152-FZ. Transmission to external commercial APIs is legally impermissible. Local execution is not a compromise but the only lawful option.

**Constrained decoding over post-processing:** Outlines integrates at the logit level and constructs a token mask at each decoding step from the JSON Schema finite automaton. The model cannot physically return an invalid response. This is a fundamentally different reliability guarantee compared to probabilistic guided decoding offered by external API providers.

**Decomposition over monolithic prompting:** on a 10B model, simultaneously performing fact extraction and narrative synthesis in a single prompt results in information loss. Decomposition into discrete cognitive subtasks allows the model to perform each step reliably.

---

## Limitations and Future Work

The system was validated on 6 EMR texts due to computational and time constraints. Three directions for further development are identified:

1. **Structured data layer.** The formalized patient clinical representation produced by the extraction stage can be extended with medical ontologies to construct knowledge graphs covering diagnoses, laboratory results, vital signs, and medication lists.

2. **Agentic feedback loop.** LangGraph already underlies both pipelines. The natural next step is to close the loop: the evaluator becomes an iterative corrector that passes structured feedback on omissions back to the summarizer, converting the static pipeline into a self-correcting system.

3. **Broader validation.** Expansion to more EMR texts, additional CT modalities, and expert annotation for ground-truth comparison.

---

## Citation

If you use this work, please reference the hackathon submission:

> Немченко И.А., Калмыкова В.И., Абдулова А.Р., Кириллов Е.О., Куправа Т.Ш. «Разработка методов генерации и автоматизированной оценки качества суммаризации медицинских текстов». Хакатон Арена LLM, Сеченовский Университет, 2025–2026.

---

## Contact

- GitHub: [https://github.com/Fortharl/Arena_LLM](https://github.com/Fortharl/Arena_LLM)
- Email: t-limf@yandex.ru
- Telegram: @mindset_paradox
