# 🎓 Scalable Academic Policy Question-Answering System

A **retrieval-first QA system** over NUST Academic Handbooks (UG & PG) using Big Data techniques. This project implements and compares **exact** (TF-IDF) and **approximate** (MinHash LSH, SimHash) similarity methods for efficient document retrieval, with a polished Streamlit web interface.

> **This is NOT a chatbot.** It is a principled, scalable retrieval system where the chatbot interface is just the final layer.

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NUST Academic Policy QA                       │
├──────────┬──────────────────────────────────────────────────────┤
│          │                                                      │
│  PDF     │  Ingestion → Chunking → Preprocessing → Indexing     │
│  (ug.pdf)│       ↓          ↓           ↓            ↓          │
│  (pg.pdf)│  pdfplumber  200-500w    Lemmatize    TF-IDF (exact) │
│          │              overlap     Synonyms     MinHash+LSH    │
│          │              sections    Shingles     SimHash         │
│          │                                                      │
│  Query ──┤→ Preprocess → Hybrid Retrieval (RRF) → PageRank     │
│          │                    ↓                     Boost        │
│          │             Answer Generator                          │
│          │           (Extractive / LLM)                          │
│          │                    ↓                                  │
│          │            Streamlit Interface                        │
└──────────┴──────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Interface
```bash
streamlit run app.py
```

### 3. Run Experiments (Evaluation)
```bash
python experiments.py
```

## 🧩 System Components

### Data Ingestion (`src/ingestion.py`)
- Parses UG and PG handbooks using `pdfplumber`
- Smart chunking: 200-500 words with 50-word overlap
- Preserves page numbers, section headers, and table data

### Text Preprocessing (`src/preprocessing.py`)
- NLTK tokenization and lemmatization
- Domain-specific synonym expansion (GPA↔CGPA, fail↔failure, etc.)
- Word shingle generation (w=3) for MinHash

### Retrieval Methods

| Method | Type | Library | Description |
|--------|------|---------|-------------|
| **TF-IDF** | Exact | scikit-learn | Cosine similarity baseline with bigrams |
| **MinHash + LSH** | Approximate | datasketch | Shingle-based Jaccard similarity with LSH indexing |
| **SimHash** | Approximate | Custom | MD5 token hashing + TF-IDF weighted fingerprints + Hamming distance |
| **Hybrid (RRF)** | Fusion | Custom | Reciprocal Rank Fusion of all three methods |

### PageRank Extension (`src/pagerank.py`)
- Builds a directed graph of handbook sections from cross-references
- Computes PageRank scores to boost retrieval results from important sections
- Uses `networkx` for graph analysis

### Answer Generation (`src/answer_generator.py`)
- **Extractive mode** (default): Sentence-level TF-IDF ranking
- **LLM mode** (optional): OpenAI GPT-4o-mini with grounded prompting
- All answers are constrained to retrieved content

## 📊 Evaluation

### Experiments Run
1. **Precision@k / Recall@k** — 15 ground truth queries × 4 methods
2. **Query Latency** — Averaged over multiple runs per method
3. **Memory Usage** — RSS/VMS tracking via psutil
4. **Parameter Sensitivity** — MinHash num_perm, LSH threshold, SimHash Hamming distance
5. **Scalability** — Corpus scaled 1x, 2x, 5x, 10x

### Sample Queries
- "What is the minimum GPA requirement?"
- "What happens if a student fails a course?"
- "What is the attendance policy?"
- "How many times can a course be repeated?"
- "What is the fee refund policy?"

## 📁 Project Structure

```
├── Handbooks/
│   ├── ug.pdf                  # UG Handbook
│   └── pg.pdf                  # PG Handbook
├── src/
│   ├── __init__.py
│   ├── ingestion.py            # PDF parsing & chunking
│   ├── preprocessing.py        # Text preprocessing & shingles
│   ├── tfidf_retriever.py      # TF-IDF exact baseline
│   ├── minhash_lsh.py          # MinHash + LSH approximate retrieval
│   ├── simhash_retriever.py    # Custom SimHash implementation
│   ├── retrieval.py            # Hybrid retrieval pipeline (RRF)
│   ├── pagerank.py             # PageRank section importance
│   ├── answer_generator.py     # Answer generation (extractive/LLM)
│   └── evaluation.py           # Metrics, plotting, sensitivity analysis
├── app.py                      # Streamlit web interface
├── experiments.py              # Full experiment runner
├── requirements.txt            # Python dependencies
├── results/                    # Generated plots & CSV results
└── README.md                   # This file
```

## 🛠 Tech Stack

- **Python 3.10+**
- **pdfplumber** — PDF parsing
- **NLTK** — NLP preprocessing
- **scikit-learn** — TF-IDF vectorization
- **datasketch** — MinHash & LSH
- **networkx** — PageRank graph analysis
- **Streamlit** — Web interface
- **Plotly** — Interactive charts
- **psutil** — Memory profiling

## 📝 Key Design Decisions

1. **Shingle-based MinHash** — Uses 3-word shingles instead of raw tokens for dramatically better Jaccard estimation with short queries
2. **Synonym Expansion** — Maps domain terms (GPA↔CGPA, fail↔failure) to bridge query-document vocabulary gap
3. **Reciprocal Rank Fusion** — Principled method to combine rankings from heterogeneous retrieval methods
4. **TF-IDF weighted SimHash** — Custom fingerprints weighted by term importance for better discrimination
5. **PageRank Boosting** — Important handbook sections get a retrieval score boost based on cross-reference graph analysis
