"""
Experiments Runner
===================
Runs all evaluations, generates plots, and saves results.
Execute: python experiments.py
"""

import os
import json
import pandas as pd
from src.ingestion import ingest_handbooks
from src.preprocessing import preprocess_dataframe
from src.retrieval import HybridRetriever
from src.evaluation import (
    evaluate_precision_recall,
    evaluate_latency,
    evaluate_memory,
    analyze_minhash_sensitivity,
    analyze_lsh_bands_sensitivity,
    analyze_simhash_sensitivity,
    scalability_test,
    generate_static_plots,
    GROUND_TRUTH_QUERIES,
)


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Data Ingestion & Preprocessing
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(" STEP 1: Data Ingestion & Preprocessing")
    print("=" * 60)

    df = ingest_handbooks()
    preprocess_dataframe(df)

    # Save processed data
    df.to_pickle(os.path.join(results_dir, "processed_chunks.pkl"))
    print(f"\n Saved {len(df)} chunks to {results_dir}/processed_chunks.pkl")

    # -----------------------------------------------------------------------
    # 2. Build Index
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STEP 2: Building Indexes")
    print("=" * 60)

    retriever = HybridRetriever()
    index_timings = retriever.fit(df)

    print("\n️  Indexing Times:")
    for key, val in index_timings.items():
        print(f"   {key}: {val:.1f} ms")

    stats = retriever.get_stats()
    print(f"\n System Stats:")
    for key, val in stats.items():
        print(f"   {key}: {val}")

    # -----------------------------------------------------------------------
    # 3. Precision & Recall Evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STEP 3: Precision & Recall Evaluation")
    print("=" * 60)

    eval_df = evaluate_precision_recall(retriever)
    eval_df.to_csv(os.path.join(results_dir, "precision_recall.csv"), index=False)

    # Summary
    summary = eval_df.groupby("method").agg({
        "precision": "mean",
        "recall": "mean",
    }).round(3)
    print("\n Average Precision & Recall:")
    print(summary.to_string())

    # -----------------------------------------------------------------------
    # 4. Latency Evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("️  STEP 4: Latency Evaluation")
    print("=" * 60)

    latency_df = evaluate_latency(retriever)
    latency_df.to_csv(os.path.join(results_dir, "latency.csv"), index=False)

    lat_summary = latency_df.groupby("method")["avg_latency_ms"].mean().round(2)
    print("\n️  Average Latency (ms):")
    print(lat_summary.to_string())

    # -----------------------------------------------------------------------
    # 5. Memory Evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STEP 5: Memory Evaluation")
    print("=" * 60)

    mem = evaluate_memory(retriever)
    print(f"   RSS: {mem['rss_mb']:.1f} MB")
    print(f"   VMS: {mem['vms_mb']:.1f} MB")

    # -----------------------------------------------------------------------
    # 6. Parameter Sensitivity Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STEP 6: Parameter Sensitivity Analysis")
    print("=" * 60)

    print("\n MinHash num_perm sensitivity:")
    minhash_sens_df = analyze_minhash_sensitivity(df)
    minhash_sens_df.to_csv(os.path.join(results_dir, "minhash_sensitivity.csv"), index=False)

    print("\n LSH threshold sensitivity:")
    lsh_sens_df = analyze_lsh_bands_sensitivity(df)
    lsh_sens_df.to_csv(os.path.join(results_dir, "lsh_sensitivity.csv"), index=False)

    print("\n SimHash Hamming threshold sensitivity:")
    simhash_sens_df = analyze_simhash_sensitivity(df)
    simhash_sens_df.to_csv(os.path.join(results_dir, "simhash_sensitivity.csv"), index=False)

    # -----------------------------------------------------------------------
    # 7. Scalability Test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STEP 7: Scalability Test")
    print("=" * 60)

    scale_df = scalability_test(df)
    scale_df.to_csv(os.path.join(results_dir, "scalability.csv"), index=False)

    # -----------------------------------------------------------------------
    # 8. Generate Plots
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STEP 8: Generating Plots")
    print("=" * 60)

    generate_static_plots(
        results_dir=results_dir,
        eval_df=eval_df,
        latency_df=latency_df,
        scalability_df=scale_df,
    )

    # -----------------------------------------------------------------------
    # 9. Qualitative Test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STEP 9: Qualitative Test (Sample Queries)")
    print("=" * 60)

    from src.answer_generator import AnswerGenerator
    generator = AnswerGenerator(mode="extractive")

    for q in GROUND_TRUTH_QUERIES[:5]:
        print(f"\n Query: {q['query']}")
        result = retriever.retrieve(q["query"], method="hybrid", top_k=5)
        answer = generator.generate(q["query"], result["results"])
        print(f" Answer ({answer['method']}): {answer['answer'][:300]}...")
        print(f"   Top chunk: {result['results'][0]['chunk_id']} (score: {result['results'][0]['score']:.4f})")
        if result["results"][0].get("page_start"):
            print(f"   Source: {result['results'][0].get('source', 'N/A')} p.{result['results'][0].get('page_start')}")

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    summary_data = {
        "num_chunks": len(df),
        "index_timings": index_timings,
        "system_stats": {k: str(v) for k, v in stats.items()},
        "memory": mem,
        "avg_precision": summary.to_dict(),
        "avg_latency": lat_summary.to_dict(),
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 60)
    print(" ALL EXPERIMENTS COMPLETE")
    print(f" Results saved to {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
