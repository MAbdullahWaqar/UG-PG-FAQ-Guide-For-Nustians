"""
Automated Evaluation Script for 200-Question Pool
=================================================
Runs the retrieval pipeline on all 200 sample questions (including variations)
and checks if the expected keywords are present in the retrieved chunks.
Reports Precision and Recall per category.
"""

import os
import json
import time
import pandas as pd
from src.retrieval import HybridRetriever
from src.ingestion import ingest_handbooks
from src.preprocessing import preprocess_dataframe

def _is_relevant(chunk_text: str, keywords: list[str], threshold: int = 1) -> bool:
    """Check if a chunk is relevant based on keyword overlap."""
    if not keywords:
        return True
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= threshold

def run_evaluation(pool_file: str = "sample_questions.json"):
    print("=" * 60)
    print("🧪 Robust Evaluation: 200-Question Pool")
    print("=" * 60)

    if not os.path.exists(pool_file):
        print(f"Error: {pool_file} not found.")
        return

    with open(pool_file, 'r') as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {pool_file}")

    # Prepare retriever
    print("Initializing pipeline...")
    df = ingest_handbooks()
    if df.empty:
        print("Failed to load handbooks.")
        return
    preprocess_dataframe(df)

    retriever = HybridRetriever()
    retriever.fit(df)

    results = []
    category_stats = {}
    total_time = 0

    print("Running evaluation (this may take a minute)...")
    for q in questions:
        # Run original question
        t0 = time.time()
        res = retriever.retrieve(q["question"], method="hybrid", top_k=5)
        latency = (time.time() - t0) * 1000
        total_time += latency

        retrieved_chunks = [r.get("text", "") for r in res["results"]]
        
        # Determine relevance
        relevant_count = 0
        for text in retrieved_chunks:
            if _is_relevant(text, q["expected_keywords"]):
                relevant_count += 1
                
        precision = relevant_count / max(len(res["results"]), 1)
        
        cat = q["category"]
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "precision_sum": 0, "failures": []}
            
        category_stats[cat]["count"] += 1
        category_stats[cat]["precision_sum"] += precision

        # Check variations (simulated user typos/paraphrases)
        var_precision_sum = 0
        for var in q["variations"]:
            var_res = retriever.retrieve(var, method="hybrid", top_k=5)
            var_relevant = sum(1 for r in var_res["results"] if _is_relevant(r.get("text", ""), q["expected_keywords"]))
            var_precision_sum += var_relevant / max(len(var_res["results"]), 1)

        avg_var_precision = var_precision_sum / len(q["variations"]) if q["variations"] else 0

        if precision == 0:
            category_stats[cat]["failures"].append(q["question"])

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": cat,
            "precision": precision,
            "variation_precision": avg_var_precision
        })

    # Print Report
    print("\n" + "=" * 60)
    print("📊 Evaluation Report")
    print("=" * 60)
    print(f"Total Questions Evaluated: {len(questions)}")
    print(f"Average Query Latency: {total_time / len(questions):.1f} ms")
    
    total_precision = sum(r["precision"] for r in results) / len(results)
    total_var_precision = sum(r["variation_precision"] for r in results) / len(results)
    
    print(f"\nOverall Precision@5: {total_precision:.3f}")
    print(f"Overall Variation Precision@5 (with typos): {total_var_precision:.3f}")

    print("\n📈 Performance by Category:")
    for cat, stats in category_stats.items():
        avg_p = stats["precision_sum"] / stats["count"]
        print(f"  - {cat.ljust(15)}: {avg_p:.3f} (n={stats['count']})")
        
    print("\n⚠️  Sample Failures (Zero relevant chunks found):")
    failures_shown = 0
    for cat, stats in category_stats.items():
        for f in stats["failures"]:
            print(f"  [X] {f}")
            failures_shown += 1
            if failures_shown >= 5:
                break
        if failures_shown >= 5:
            break

    if failures_shown == 0:
        print("  None! All questions retrieved at least one relevant chunk.")

    # Save detailed results
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/robust_evaluation_results.csv", index=False)
    print("\nDetailed results saved to results/robust_evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()
