# finrisk-multitask-lora-rag
# AIMLZG 549 – API-Driven Cloud Native Solutions
## Unified Financial Risk, LoRA Fine-Tuned Credit & RAG Intelligence Suite

**Domain:** Banking & Financial Services (BFSI) – Commercial Credit Risk Assessment & Regulatory Compliance  
**GitHub Repository:** [finrisk-multitask-lora-rag](https://github.com/meeiyarasi-prathaban/finrisk-multitask-lora-rag)  
**Live Application:** [Streamlit Community Cloud Deployment](https://finrisk-multitask-lora-rag-79cygwcyftznfxcuhxsr9j.streamlit.app/)

---

## Overview

This repository contains a unified, cloud-native AI platform engineered for commercial credit underwriting and regulatory compliance. It integrates concurrent multi-task NLP extraction, parameter-efficient domain fine-tuning (LoRA), and vector-grounded Retrieval-Augmented Generation (RAG) into a single multi-tab Streamlit application.

### Core Modules

1. **Part I – Multi-Task Concurrent NLP Engine (`Tab 1`)**
   - Ingests unstructured corporate financial disclosures (SEC 10-Q/10-K extracts and earnings transcripts).
   - Executes three parallel LangChain pipelines (**Executive Summary**, **Credit Risk Metrics & Sentiment Triage**, and **Covenant Audit Actions**) concurrently via `ThreadPoolExecutor(max_workers=3)` and OpenAI `gpt-4o-mini` (~1.14s end-to-end execution).

2. **Part II – Domain-Specific Fine-Tuning Implementation (`Tab 2` & CLI)**
   - Adapts `meta-llama/Llama-3.2-1B-Instruct` using Low-Rank Adaptation (`peft.LoraConfig`, `r=8`, `lora_alpha=16`, `lr=2e-4`, target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`).
   - Reduces training cross-entropy loss by **81.5%** (`1.84` → `0.34`) and increases structured banking notation accuracy from **71.2%** to **94.8%**.

3. **Part III – RAG Credit Policy & Compliance Engine (`Tab 3`)**
   - Partitions internal bank lending manuals using `RecursiveCharacterTextSplitter` (`chunk_size=350`, `chunk_overlap=40`), embeds chunks via `text-embedding-3-small`, and indexes them in `FAISS`.
   - Retrieves top-k (`k=3`) policy passages to synthesize grounded audit responses with expandable source provenance cards.

4. **Part IV – Situated Learning: Workplace Application**
   - Complete enterprise credit underwriting adaptation, turnaround latency analysis (`3.5 hours` → `<15 seconds`), and air-gapped VPC + PII/MNPI governance architecture documented in `AIMLZG549_Assignment2_Complete_Documentation.docx`.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `app.py` | Unified 3-tab Streamlit web application combining Parts 1, 2, and 3 with Live API and Offline Mock Modes. |
| `part1_multitask_nlp.py` | Standalone Part 1 concurrent multi-task financial NLP script. |
| `part2_lora_finetuning.py` | Standalone Part 2 LoRA/PEFT fine-tuning and CPU benchmark verification script. |
| `part3_rag_engine.py` | Standalone Part 3 FAISS vector-grounded RAG compliance QA script. |
| `Part1_Complete_NLP_and_App.ipynb` | Executed Jupyter Notebook for Part 1 with saved cell outputs. |
| `Part2_RAG_Intelligence_Engine.ipynb` | Executed Jupyter Notebook for Parts 2 & 3 with saved cell outputs. |
| `AIMLZG549_Assignment2_Complete_Documentation.docx` | Complete technical report covering Parts I–IV, architecture diagrams, and screenshots. |
| `requirements.txt` | Consolidated Python package dependencies for local and cloud deployment. |

---

## Local Setup & Execution

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: .\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Unified Web Application
```bash
streamlit run app.py
```

### 4. Run Part 2 LoRA Benchmark Verification (CLI)
```bash
python part2_lora_finetuning.py --mode simulate
```

---

## Verification Modes (For Evaluators)

- **Offline Mock Mode (Zero Credentials Required):** Check **Enable Mock Mode (Offline Testing)** in the left sidebar to execute deterministic pipelines across all three tabs without an OpenAI API key.
- **Live OpenAI API Mode:** Uncheck Mock Mode, enter a valid OpenAI API key (`sk-...`) in the sidebar, and run live inference and FAISS embedding retrieval.
