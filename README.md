# 🎓 Scalable Academic Policy QA System (Big Data Architecture)

This project is a **principled, highly scalable Question-Answering (QA) system** designed to retrieve and answer queries based on the NUST Undergraduate (UG) and Postgraduate (PG) Handbooks. 

> **Important Note:** This project is NOT merely a chatbot wrapper around an LLM. It is a ground-up implementation of Big Data retrieval techniques, focusing on the tradeoff between exact and approximate similarity search at scale. The user interface (Streamlit) and LLM synthesis are simply the final presentation layers of a robust mathematical pipeline.

---

##  1. System Pipeline Overview

```text
┌───────────────────────────────────────────────────────────────────┐
│                   NUST Academic Policy QA Pipeline                │
├───────────┬───────────────────────────────────────────────────────┤
│           │                                                       │
│ 1. Input  │  UG Handbook (PDF)          PG Handbook (PDF)         │
│           │        │                           │                  │
│           │        └────────────┬──────────────┘                  │
│           │                     ↓                                 │
│ 2. Parse  │     Extract Text & Tables (pdfplumber)                │
│           │     Chunking (200-500 words, 50-word overlap)         │
│           │                     ↓                                 │
│ 3. Clean  │     Tokenize, Lemmatize, Synonym Expansion (NLTK)     │
│           │     Generate 3-word Shingles                          │
│           │                     ↓                                 │
│ 4. Index  │ ┌──────────────┬────────────────┬─────────────────┐   │
│           │ │ Exact Match  │  Approximate   │   Approximate   │   │
│           │ │    TF-IDF    │  MinHash LSH   │    SimHash      │   │
│           │ │ (Scikit)     │ (Datasketch)   │  (Bitwise MD5)  │   │
│           │ └──────┬───────┴────────┬───────┴─────────┬───────┘   │
│           │        │                │                 │           │
│ 5. Query  │ ┌──────┴────────────────┴─────────────────┴───────┐   │
│           │ │      Reciprocal Rank Fusion (Hybrid RRF)        │   │
│           │ │        + PageRank Importance Boosting           │   │
│           │ └───────────────────────┬─────────────────────────┘   │
│           │                         ↓                             │
│ 6. Output │ ┌───────────────────────┴─────────────────────────┐   │
│           │ │                Answer Generator                 │   │
│           │ │   [Extractive Mode]        [LLM API]   │   │
│           │ └───────────────────────┬─────────────────────────┘   │
│           │                         ↓                             │
│           │                 Streamlit Web UI                      │
└───────────┴───────────────────────────────────────────────────────┘
```

The system operates through a sequential, data-intensive pipeline designed to handle large text corpora efficiently:

1. **Data Ingestion:** Raw PDFs are parsed, cleaned, and split into semantically meaningful chunks (200-500 words).
2. **Preprocessing:** Text undergoes NLP normalization (lemmatization, stop-word removal, domain-specific synonym expansion, and n-gram shingling).
3. **Indexing & Hashing:** Documents are indexed using multiple Big Data paradigms (Exact TF-IDF, Approximate MinHash, Approximate SimHash).
4. **Query Retrieval:** A user query is processed and evaluated against the indexes to find the top-$K$ most relevant chunks.
5. **PageRank Boosting (Extension):** Retrieved chunks are re-ranked based on the structural importance of their source section (calculated via PageRank on handbook cross-references).
6. **Answer Generation:** The context is fed into either a purely Extractive algorithm or a Cloud LLM (Groq Llama 3) to format the final answer.

---

##  2. Deep Dive: Retrieval Methods

The core of this project is the implementation and comparison of three distinct retrieval architectures.

### A. TF-IDF + Cosine Similarity (The Exact Baseline)
* **How it works:** This is the non-approximate baseline. Every document chunk is represented as a sparse vector in a high-dimensional space where each dimension represents a unique word/bigram. The value is calculated using Term Frequency-Inverse Document Frequency (TF-IDF).
* **Retrieval:** When a query is made, it is vectorized into the same space. The system calculates the Cosine Similarity (the dot product normalized by vector magnitudes) between the query and every document.
* **Pros & Cons:** Extremely accurate (high precision), but computationally expensive ($O(N)$ time complexity for $N$ documents), making it unscalable for massive datasets.

### B. MinHash + Locality Sensitive Hashing (LSH) (Approximate)
* **How it works:** Documents are treated as Sets of 3-word "shingles". To avoid comparing massive sets, we use MinHash to generate a small, fixed-size mathematical "signature" (e.g., 128 hash permutations) for each document. The probability that two documents share the same MinHash value is equal to their Jaccard Similarity.
* **LSH Indexing:** To avoid comparing the query signature against every document signature ($O(N)$), we use Locality Sensitive Hashing (LSH). The 128-hash signature is split into "bands". If two documents match in at least one band, they are placed in the same hash bucket. 
* **Retrieval:** At query time, we only calculate Jaccard similarity against documents in the same LSH bucket ($O(1)$ lookup time).
* **Pros & Cons:** Massively scalable and fast. However, it trades off slight accuracy (it might miss a marginally similar document that didn't hash into the same bucket).

### C. SimHash (Approximate)
* **How it works:** Unlike MinHash which uses sets, SimHash generates a single 64-bit fingerprint for a document. It works by taking the MD5 hash of every word, weighting the bits (+1 for a '1', -1 for a '0') by the word's TF-IDF importance, and summing them up. If the final sum for a bit position is positive, the fingerprint gets a 1; otherwise, a 0.
* **Retrieval:** Similarity is measured using **Hamming Distance** (the number of differing bits). Documents with a Hamming distance below a certain threshold are returned.
* **Pros & Cons:** Extremely memory efficient (just 64 bits per document) and fast bitwise XOR comparisons. However, it struggles with very short queries compared to MinHash.

### D. Reciprocal Rank Fusion (RRF) - The Hybrid Approach
To get the best of both worlds, the system implements **RRF**. It runs TF-IDF, MinHash, and SimHash simultaneously. Instead of comparing raw scores (which are on different mathematical scales), it ranks the documents by position ($1^{st}, 2^{nd}, 3^{rd}$) and calculates a combined fusion score: 
$$RRF Score = \sum \frac{1}{k + rank}$$
This ensures the final output is highly robust.

---

##  3. Competitive Edge Extensions

To stand out, this project implements two advanced Big Data concepts:

### PageRank for Section Importance
Not all policies in a handbook are equally important. By parsing cross-references (e.g., *"As mentioned in Section 4.1..."*), the system builds a **Directed Graph** of the handbook using `networkx`. We apply the **PageRank algorithm** to determine which sections are the most "authoritative". 
During retrieval, chunks from high-PageRank sections receive a mathematical score boost, ensuring core academic policies surface before minor footnotes.

### Generative vs. Extractive Modes & Auto-Fallback Cascade
* **Retrieval Based (Extractive):** Uses sentence-level TF-IDF to extract the exact sentences from the handbook and format them into UG and PG categories. Zero risk of hallucination.
* **Groq API (Generative):** Uses Cloud LLMs via the Groq API. It reads the retrieved context and synthesizes a human-readable answer.
  * **Auto-Fallback Cascade Architecture:** To handle strict free-tier rate limits, the system implements an automatic model cascade. It first attempts to use the massive `llama-3.3-70b-versatile` model. If a Rate Limit (`429`) or Connection Error occurs, it gracefully catches the exception and falls back to `llama-3.1-8b-instant`, then `gemma2-9b-it`, then `mixtral-8x7b-32768`. If all Cloud models fail, it instantly defaults back to the Extractive method, ensuring the application **never crashes**.
  * **Manual Override:** Users can manually select specific Groq models from the sidebar to bypass the cascade or save tokens on larger models.
---

##  4. Experimental Results & Analysis

The project includes an `experiments.py` suite that automatically runs and generates plots for the following metrics:

1. **Exact vs Approximate Tradeoffs:** 
   * **Accuracy:** TF-IDF consistently hits 100% precision. MinHash achieves ~85-95% precision, while SimHash sits around 75%.
   * **Latency:** MinHash LSH is heavily optimized, often retrieving chunks in under 2ms, whereas TF-IDF takes linearly longer as the dataset grows.
2. **Parameter Sensitivity:** 
   * Increasing the number of MinHash permutations (e.g., 64 -> 256) increases accuracy but uses more memory.
   * Adjusting the LSH threshold (e.g., 0.1 vs 0.5) wildly changes how many "candidate" documents are returned.
3. **Scalability:** We simulate scaling the corpus by 10x. TF-IDF query time degrades linearly, while MinHash LSH query time remains nearly constant ($O(1)$ bucket lookups).

*(All results, graphs, and CSVs are generated in the `/results` folder).*

---

##  5. How to Run the Project

### Prerequisites
* Python 3.10 or higher
* Mac, Linux, or Windows

### 1. Install Dependencies
```bash
# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Keys (Optional but Recommended)
To use the LLM synthesis feature, you need a free Groq API key.
```bash
export GROQ_API_KEY="your_api_key_here"
```
*(If you don't provide a key, the system seamlessly falls back to the native Extractive Retrieval mode).*

### 3. Run the Web Interface
Start the Streamlit application to interact with the QA system, view the evaluation dashboard, and explore the PageRank graphs.
```bash
streamlit run app.py
```

### 4. Run the Experiments Suite
To reproduce the Big Data tradeoff analysis, run the experiments script. This will process the handbooks, calculate metrics, and generate performance graphs in the `/results` folder.
```bash
python experiments.py
```

---

## 📁 6. Project Structure

```text
├── Handbooks/
│   ├── ug.pdf                  # Undergraduate Handbook
│   └── pg.pdf                  # Postgraduate Handbook
├── src/
│   ├── ingestion.py            # PDF parsing & text chunking
│   ├── preprocessing.py        # Tokenization, Lemmatization, Shingling
│   ├── tfidf_retriever.py      # TF-IDF Exact Baseline
│   ├── minhash_lsh.py          # MinHash + LSH Approximate
│   ├── simhash_retriever.py    # SimHash Fingerprinting
│   ├── retrieval.py            # Hybrid RRF logic
│   ├── pagerank.py             # Graph building and PageRank scoring
│   ├── answer_generator.py     # Extractive & Groq synthesis logic
│   └── evaluation.py           # Robust evaluation metrics (Precision/Recall)
├── app.py                      # Main Streamlit Web Application
├── experiments.py              # CLI script to run all Big Data experiments
├── requirements.txt            # Python dependencies
└── results/                    # Output directory for CSVs and plots
```

---
*Developed for the CS-404 Big Data Course Project.*
