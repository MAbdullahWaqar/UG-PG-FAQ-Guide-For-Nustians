import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.ingestion import ingest_handbooks
from src.preprocessing import preprocess_chunks
from src.evaluation import Evaluator

def main():
    print("Loading / Ingesting data...")
    if os.path.exists("processed_chunks.csv"):
        df = pd.read_csv("processed_chunks.csv")
    else:
        df_raw = ingest_handbooks(chunk_size=200)
        df = preprocess_chunks(df_raw)
        df.to_csv("processed_chunks.csv", index=False)
        
    print(f"Total chunks: {len(df)}")
    
    evaluator = Evaluator(df)
    
    test_queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?"
    ]
    
    print("\n--- 1. Evaluating Exact vs Approx Methods ---")
    results_df = evaluator.evaluate_queries(test_queries, top_k=5)
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv("evaluation_results.csv", index=False)
    
    # Plotting Recall
    plt.figure(figsize=(10, 6))
    # using errorbar plotting without ci parameter, since ci is deprecated in newer seaborn
    sns.barplot(data=results_df, x='Method', y='Recall@k', errorbar=None)
    plt.title('Average Recall@5 (Using TF-IDF as Ground Truth)')
    plt.savefig('recall_plot.png')
    
    print("\n--- 2. Scalability Test ---")
    # Test with 1x, 2x, 5x size
    scale_df = evaluator.run_scalability_test(test_queries[0], multipliers=[1, 2, 5])
    print(scale_df.to_string(index=False))
    
    scale_df.to_csv("scalability_results.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scale_df, x='Corpus Multiplier', y='Latency (s)', hue='Method', marker='o')
    plt.title('Query Latency vs Corpus Size')
    plt.savefig('scalability_plot.png')
    
    print("\nExperiments complete. Results saved to CSVs and plots to PNGs.")

if __name__ == "__main__":
    main()
