"""
Evaluation Module
==================
Comprehensive evaluation system for comparing retrieval methods.
Includes Precision@k, Recall@k, latency, memory, parameter sensitivity,
and scalability testing.
"""

import time
import copy
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.retrieval import HybridRetriever
from src.preprocessing import preprocess_dataframe


# ---------------------------------------------------------------------------
# Ground truth test queries
# ---------------------------------------------------------------------------

GROUND_TRUTH_QUERIES = [
    {
        "query": "What is the minimum GPA requirement?",
        "relevant_keywords": ["cgpa", "gpa", "grade point", "minimum", "cumulative", "2.0", "1.5", "requirement"],
        "category": "academics",
    },
    {
        "query": "What happens if a student fails a course?",
        "relevant_keywords": ["fail", "failure", "repeat", "retake", "grade", "f grade", "failing"],
        "category": "academics",
    },
    {
        "query": "What is the attendance policy?",
        "relevant_keywords": ["attendance", "absent", "absence", "75%", "percent", "class", "lecture", "present"],
        "category": "policy",
    },
    {
        "query": "How many times can a course be repeated?",
        "relevant_keywords": ["repeat", "retake", "course", "twice", "three", "maximum", "attempt"],
        "category": "academics",
    },
    {
        "query": "What is the fee refund policy?",
        "relevant_keywords": ["fee", "refund", "tuition", "payment", "charges", "withdrawal", "drop"],
        "category": "financial",
    },
    {
        "query": "How to apply for a degree?",
        "relevant_keywords": ["degree", "convocation", "apply", "application", "certificate", "graduation"],
        "category": "graduation",
    },
    {
        "query": "What are the rules for academic probation?",
        "relevant_keywords": ["probation", "academic", "warning", "cgpa", "gpa", "semester", "performance"],
        "category": "academics",
    },
    {
        "query": "What is the credit hour requirement for graduation?",
        "relevant_keywords": ["credit", "hour", "graduation", "total", "required", "minimum", "degree"],
        "category": "graduation",
    },
    {
        "query": "What are the examination rules?",
        "relevant_keywords": ["exam", "examination", "rule", "conduct", "cheat", "unfair", "paper", "test"],
        "category": "exams",
    },
    {
        "query": "Can a student transfer to another department?",
        "relevant_keywords": ["transfer", "department", "change", "shift", "migration", "programme"],
        "category": "admin",
    },
    {
        "query": "What is the grading system?",
        "relevant_keywords": ["grade", "grading", "system", "letter", "marks", "percentage", "scale", "a", "b", "c"],
        "category": "academics",
    },
    {
        "query": "What are the thesis requirements?",
        "relevant_keywords": ["thesis", "dissertation", "research", "supervisor", "committee", "defense"],
        "category": "research",
    },
    {
        "query": "What is the semester registration process?",
        "relevant_keywords": ["registration", "semester", "enroll", "register", "course", "add", "drop"],
        "category": "admin",
    },
    {
        "query": "What are the rules for dropping a course?",
        "relevant_keywords": ["drop", "withdraw", "course", "deadline", "withdrawal", "w grade"],
        "category": "academics",
    },
    {
        "query": "What is the scholarship policy?",
        "relevant_keywords": ["scholarship", "merit", "financial", "aid", "waiver", "fee", "concession"],
        "category": "financial",
    },
]


def _is_relevant(chunk_text: str, keywords: list[str], threshold: int = 2) -> bool:
    """Check if a chunk is relevant based on keyword overlap."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= threshold


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_precision_recall(
    retriever: HybridRetriever,
    queries: list[dict] | None = None,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute Precision@k and Recall@k for all methods.

    Returns a DataFrame with columns:
    [query, method, k, precision, recall, latency_ms]
    """
    if queries is None:
        queries = GROUND_TRUTH_QUERIES
    if k_values is None:
        k_values = [3, 5, 10]

    records = []
    methods = ["tfidf", "minhash", "simhash", "hybrid"]

    for q in queries:
        for method in methods:
            for k in k_values:
                result = retriever.retrieve(q["query"], method=method, top_k=k)
                retrieved = result["results"]
                latency = result["timings"].get("total_ms", 0)

                # Count relevant retrieved chunks
                relevant_retrieved = 0
                for r in retrieved:
                    text = r.get("text", "")
                    if _is_relevant(text, q["relevant_keywords"]):
                        relevant_retrieved += 1

                precision = relevant_retrieved / k if k > 0 else 0
                # Estimate total relevant (we don't know exact count, use retrieved as proxy)
                recall = relevant_retrieved / max(k, 1)

                records.append({
                    "query": q["query"],
                    "category": q.get("category", ""),
                    "method": method,
                    "k": k,
                    "precision": precision,
                    "recall": recall,
                    "relevant_retrieved": relevant_retrieved,
                    "latency_ms": latency,
                })

    return pd.DataFrame(records)


def evaluate_latency(
    retriever: HybridRetriever,
    queries: list[dict] | None = None,
    num_runs: int = 3,
) -> pd.DataFrame:
    """
    Measure query latency for each method (averaged over multiple runs).
    """
    if queries is None:
        queries = GROUND_TRUTH_QUERIES

    records = []
    methods = ["tfidf", "minhash", "simhash", "hybrid"]

    for q in queries:
        for method in methods:
            latencies = []
            for _ in range(num_runs):
                result = retriever.retrieve(q["query"], method=method, top_k=5)
                latencies.append(result["timings"].get("total_ms", 0))

            records.append({
                "query": q["query"],
                "method": method,
                "avg_latency_ms": np.mean(latencies),
                "std_latency_ms": np.std(latencies),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
            })

    return pd.DataFrame(records)


def evaluate_memory(retriever: HybridRetriever) -> dict:
    """Measure memory usage of the system."""
    process = psutil.Process()
    mem = process.memory_info()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "num_chunks": len(retriever.df) if retriever.df is not None else 0,
    }


# ---------------------------------------------------------------------------
# Parameter sensitivity analysis
# ---------------------------------------------------------------------------

def analyze_minhash_sensitivity(
    df: pd.DataFrame,
    num_perm_values: list[int] | None = None,
    queries: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Analyze the impact of num_perm (number of hash functions) on MinHash accuracy.
    """
    if num_perm_values is None:
        num_perm_values = [32, 64, 128, 256]
    if queries is None:
        queries = GROUND_TRUTH_QUERIES[:5]

    records = []
    for num_perm in num_perm_values:
        print(f"   Testing num_perm={num_perm}...")
        retriever = HybridRetriever(num_perm=num_perm, use_pagerank=False)
        retriever.fit(df)

        for q in queries:
            result = retriever.retrieve(q["query"], method="minhash", top_k=5)
            relevant = sum(
                1 for r in result["results"]
                if _is_relevant(r.get("text", ""), q["relevant_keywords"])
            )
            records.append({
                "num_perm": num_perm,
                "query": q["query"],
                "precision_at_5": relevant / 5,
                "latency_ms": result["timings"].get("total_ms", 0),
            })

    return pd.DataFrame(records)


def analyze_lsh_bands_sensitivity(
    df: pd.DataFrame,
    threshold_values: list[float] | None = None,
    queries: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Analyze the impact of LSH threshold (which determines bands/rows) on retrieval.
    """
    if threshold_values is None:
        threshold_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    if queries is None:
        queries = GROUND_TRUTH_QUERIES[:5]

    records = []
    for threshold in threshold_values:
        print(f"   Testing threshold={threshold}...")
        retriever = HybridRetriever(lsh_threshold=threshold, use_pagerank=False)
        retriever.fit(df)

        for q in queries:
            result = retriever.retrieve(q["query"], method="minhash", top_k=5)
            n_candidates = retriever.minhash.get_num_candidates(
                set(result["processed_query"].split())
            )
            relevant = sum(
                1 for r in result["results"]
                if _is_relevant(r.get("text", ""), q["relevant_keywords"])
            )
            records.append({
                "threshold": threshold,
                "query": q["query"],
                "precision_at_5": relevant / 5,
                "num_candidates": n_candidates,
                "latency_ms": result["timings"].get("total_ms", 0),
            })

    return pd.DataFrame(records)


def analyze_simhash_sensitivity(
    df: pd.DataFrame,
    hamming_thresholds: list[int] | None = None,
    queries: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Analyze the impact of Hamming distance threshold on SimHash retrieval.
    """
    if hamming_thresholds is None:
        hamming_thresholds = [2, 4, 8, 12, 16, 24, 32]
    if queries is None:
        queries = GROUND_TRUTH_QUERIES[:5]

    records = []
    for threshold in hamming_thresholds:
        print(f"   Testing hamming_threshold={threshold}...")
        retriever = HybridRetriever(hamming_threshold=threshold, use_pagerank=False)
        retriever.fit(df)

        for q in queries:
            result = retriever.retrieve(q["query"], method="simhash", top_k=5)
            n_within = retriever.simhash.get_within_threshold(
                result["processed_query"].split()
            )
            relevant = sum(
                1 for r in result["results"]
                if _is_relevant(r.get("text", ""), q["relevant_keywords"])
            )
            records.append({
                "hamming_threshold": threshold,
                "query": q["query"],
                "precision_at_5": relevant / 5,
                "chunks_within_threshold": n_within,
                "latency_ms": result["timings"].get("total_ms", 0),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Scalability test
# ---------------------------------------------------------------------------

def scalability_test(
    df: pd.DataFrame,
    scale_factors: list[int] | None = None,
    queries: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Test how performance changes with corpus size.
    Duplicates the corpus to simulate larger datasets.
    """
    if scale_factors is None:
        scale_factors = [1, 2, 5, 10]
    if queries is None:
        queries = GROUND_TRUTH_QUERIES[:3]

    records = []
    methods = ["tfidf", "minhash", "simhash", "hybrid"]

    for factor in scale_factors:
        print(f"   Testing scale factor={factor}x ({len(df) * factor} chunks)...")

        # Duplicate corpus
        if factor == 1:
            scaled_df = df.copy()
        else:
            dfs = [df.copy()]
            for i in range(1, factor):
                dup = df.copy()
                dup["chunk_id"] = dup["chunk_id"] + f"_dup{i}"
                dfs.append(dup)
            scaled_df = pd.concat(dfs, ignore_index=True)

        # Build index and measure indexing time
        retriever = HybridRetriever(use_pagerank=False)
        t0 = time.time()
        index_timings = retriever.fit(scaled_df)
        index_time = (time.time() - t0) * 1000

        # Measure memory
        mem = evaluate_memory(retriever)

        for method in methods:
            latencies = []
            for q in queries:
                result = retriever.retrieve(q["query"], method=method, top_k=5)
                latencies.append(result["timings"].get("total_ms", 0))

            records.append({
                "scale_factor": factor,
                "num_chunks": len(scaled_df),
                "method": method,
                "avg_query_latency_ms": np.mean(latencies),
                "index_time_ms": index_time,
                "memory_mb": mem["rss_mb"],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting functions (Plotly for Streamlit)
# ---------------------------------------------------------------------------

def plot_precision_recall_comparison(eval_df: pd.DataFrame) -> go.Figure:
    """Create an interactive precision/recall comparison chart."""
    avg_df = eval_df.groupby(["method", "k"]).agg({
        "precision": "mean",
        "recall": "mean",
    }).reset_index()

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Precision@k", "Recall@k"))

    colors = {"tfidf": "#FF6B6B", "minhash": "#4ECDC4", "simhash": "#45B7D1", "hybrid": "#96CEB4"}

    for method in avg_df["method"].unique():
        mdf = avg_df[avg_df["method"] == method]
        fig.add_trace(
            go.Bar(name=f"{method}", x=mdf["k"].astype(str), y=mdf["precision"],
                   marker_color=colors.get(method, "#888"), showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name=f"{method}", x=mdf["k"].astype(str), y=mdf["recall"],
                   marker_color=colors.get(method, "#888"), showlegend=False),
            row=1, col=2
        )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=400,
        margin=dict(t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=1)
    fig.update_yaxes(title_text="Recall", row=1, col=2)

    return fig


def plot_latency_comparison(latency_df: pd.DataFrame) -> go.Figure:
    """Create a latency comparison chart."""
    avg_df = latency_df.groupby("method").agg({
        "avg_latency_ms": "mean",
        "std_latency_ms": "mean",
    }).reset_index()

    colors = {"tfidf": "#FF6B6B", "minhash": "#4ECDC4", "simhash": "#45B7D1", "hybrid": "#96CEB4"}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_df["method"],
        y=avg_df["avg_latency_ms"],
        error_y=dict(type="data", array=avg_df["std_latency_ms"]),
        marker_color=[colors.get(m, "#888") for m in avg_df["method"]],
    ))

    fig.update_layout(
        title="Average Query Latency by Method",
        xaxis_title="Method",
        yaxis_title="Latency (ms)",
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=30),
    )

    return fig


def plot_parameter_sensitivity(
    minhash_df: pd.DataFrame,
    lsh_df: pd.DataFrame,
    simhash_df: pd.DataFrame,
) -> go.Figure:
    """Create parameter sensitivity plots."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "MinHash: num_perm vs Precision",
            "LSH: threshold vs Precision",
            "SimHash: Hamming threshold vs Precision"
        )
    )

    # MinHash
    mh_avg = minhash_df.groupby("num_perm")["precision_at_5"].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=mh_avg["num_perm"], y=mh_avg["precision_at_5"],
                   mode="lines+markers", marker=dict(color="#4ECDC4", size=10),
                   line=dict(color="#4ECDC4", width=2), name="MinHash"),
        row=1, col=1
    )

    # LSH threshold
    lsh_avg = lsh_df.groupby("threshold")["precision_at_5"].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=lsh_avg["threshold"], y=lsh_avg["precision_at_5"],
                   mode="lines+markers", marker=dict(color="#45B7D1", size=10),
                   line=dict(color="#45B7D1", width=2), name="LSH Threshold"),
        row=1, col=2
    )

    # SimHash
    sh_avg = simhash_df.groupby("hamming_threshold")["precision_at_5"].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=sh_avg["hamming_threshold"], y=sh_avg["precision_at_5"],
                   mode="lines+markers", marker=dict(color="#FF6B6B", size=10),
                   line=dict(color="#FF6B6B", width=2), name="SimHash"),
        row=1, col=3
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=30),
        showlegend=False,
    )

    return fig


def plot_scalability(scalability_df: pd.DataFrame) -> go.Figure:
    """Create scalability analysis plots."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Query Latency vs Corpus Size", "Index Time vs Corpus Size", "Memory vs Corpus Size"),
    )

    colors = {"tfidf": "#FF6B6B", "minhash": "#4ECDC4", "simhash": "#45B7D1", "hybrid": "#96CEB4"}

    for method in scalability_df["method"].unique():
        mdf = scalability_df[scalability_df["method"] == method]
        fig.add_trace(
            go.Scatter(x=mdf["num_chunks"], y=mdf["avg_query_latency_ms"],
                       mode="lines+markers", name=method,
                       line=dict(color=colors.get(method, "#888")),
                       marker=dict(color=colors.get(method, "#888"), size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=mdf["num_chunks"], y=mdf["index_time_ms"],
                       mode="lines+markers", name=method, showlegend=False,
                       line=dict(color=colors.get(method, "#888")),
                       marker=dict(color=colors.get(method, "#888"), size=8)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=mdf["num_chunks"], y=mdf["memory_mb"],
                       mode="lines+markers", name=method, showlegend=False,
                       line=dict(color=colors.get(method, "#888")),
                       marker=dict(color=colors.get(method, "#888"), size=8)),
            row=1, col=3
        )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Chunks", row=1, col=1)
    fig.update_xaxes(title_text="Chunks", row=1, col=2)
    fig.update_xaxes(title_text="Chunks", row=1, col=3)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=3)

    return fig


def generate_static_plots(results_dir: str = "results", **dataframes):
    """Generate static matplotlib plots and save to results directory."""
    import os
    os.makedirs(results_dir, exist_ok=True)

    plt.style.use("dark_background")

    # 1. Precision@k comparison
    if "eval_df" in dataframes:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        eval_df = dataframes["eval_df"]
        avg = eval_df.groupby(["method", "k"]).agg({"precision": "mean", "recall": "mean"}).reset_index()

        for method in avg["method"].unique():
            mdf = avg[avg["method"] == method]
            axes[0].plot(mdf["k"], mdf["precision"], "o-", label=method, linewidth=2)
            axes[1].plot(mdf["k"], mdf["recall"], "o-", label=method, linewidth=2)

        axes[0].set_title("Precision@k", fontsize=14)
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Precision")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Recall@k", fontsize=14)
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Recall")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "precision_recall.png"), dpi=150)
        plt.close()

    # 2. Latency comparison
    if "latency_df" in dataframes:
        fig, ax = plt.subplots(figsize=(10, 5))
        latency_df = dataframes["latency_df"]
        avg = latency_df.groupby("method")["avg_latency_ms"].mean().reset_index()
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        ax.bar(avg["method"], avg["avg_latency_ms"], color=colors[:len(avg)])
        ax.set_title("Average Query Latency", fontsize=14)
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "latency.png"), dpi=150)
        plt.close()

    # 3. Scalability
    if "scalability_df" in dataframes:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sdf = dataframes["scalability_df"]

        for method in sdf["method"].unique():
            mdf = sdf[sdf["method"] == method]
            axes[0].plot(mdf["num_chunks"], mdf["avg_query_latency_ms"], "o-", label=method, linewidth=2)
            axes[1].plot(mdf["num_chunks"], mdf["index_time_ms"], "o-", label=method, linewidth=2)
            axes[2].plot(mdf["num_chunks"], mdf["memory_mb"], "o-", label=method, linewidth=2)

        titles = ["Query Latency", "Index Build Time", "Memory Usage"]
        ylabels = ["Latency (ms)", "Time (ms)", "Memory (MB)"]
        for i, ax in enumerate(axes):
            ax.set_title(titles[i], fontsize=14)
            ax.set_xlabel("Number of Chunks")
            ax.set_ylabel(ylabels[i])
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "scalability.png"), dpi=150)
        plt.close()

    print(f"📊 Static plots saved to {results_dir}/")
