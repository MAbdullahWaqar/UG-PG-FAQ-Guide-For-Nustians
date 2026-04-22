# Scalable Academic Policy Question-Answering System

This project is a scalable, retrieval-first Question-Answering (QA) system over academic handbooks. It uses Big Data techniques to parse PDFs, chunk text, and index content using both Exact (TF-IDF) and Approximate (MinHash LSH, SimHash) similarity algorithms.

## Features
- **Data Ingestion**: Parses PDFs, cleans text, and splits into manageable chunks while preserving metadata.
- **Indexing Methods**: 
  - `TF-IDF + Cosine Similarity` (Exact match baseline)
  - `MinHash + LSH` (Approximate similarity using Jaccard distance)
  - `SimHash` (Custom implementation of Charikar's SimHash using Hamming distance)
- **Answer Generation**: Extractive answering that finds the most relevant sentences from the top retrieved chunks.
- **Recommendation/Re-ranking Extension**: Re-ranks retrieved chunks based on their position in the original document combined with their similarity score.
- **Streamlit Interface**: Interactive UI to query the system and observe how different retrieval methods perform.

## Project Structure
```text
project/
├── Handbooks/               # Raw PDF documents
├── src/
│   ├── ingestion.py         # PDF parsing and chunking
│   ├── preprocessing.py     # NLTK tokenization and lemmatization
│   ├── tfidf_baseline.py    # Exact similarity retriever
│   ├── minhash_lsh.py       # Approximate similarity retriever
│   ├── simhash.py           # Custom SimHash implementation
│   ├── retrieval.py         # Unified query processing and retrieval
│   ├── answer_generation.py # Extractive answer logic
│   └── evaluation.py        # Precision/Recall and scalability metrics
├── app.py                   # Streamlit user interface
├── experiments.py           # Script to run metrics and scalability tests
└── requirements.txt         # Project dependencies
```

## Setup & Installation

1. Create a virtual environment (optional but recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Interface
To launch the interactive QA app:
```bash
streamlit run app.py
```
*(On first run, the system will ingest the PDFs, process them, and build the indices. This may take a minute or two.)*

### 2. Run Experiments & Evaluation
To run the evaluation queries, compare Precision/Recall against the TF-IDF baseline, and perform scalability tests:
```bash
python experiments.py
```
This script will output `recall_plot.png` and `scalability_plot.png` to visualize the performance of the exact vs. approximate methods.
