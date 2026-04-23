"""
Streamlit Web Interface
========================
Premium dark-themed QA interface with glassmorphism design,
method comparison, evaluation dashboards, and PageRank visualization.
"""

import os
import time
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ingestion import ingest_handbooks
from src.preprocessing import preprocess_dataframe
from src.retrieval import HybridRetriever
from src.answer_generator import AnswerGenerator
from src.evaluation import (
    evaluate_precision_recall,
    evaluate_latency,
    evaluate_memory,
    analyze_minhash_sensitivity,
    analyze_lsh_bands_sensitivity,
    analyze_simhash_sensitivity,
    scalability_test,
    plot_precision_recall_comparison,
    plot_latency_comparison,
    plot_parameter_sensitivity,
    plot_scalability,
    GROUND_TRUTH_QUERIES,
)


# ═══════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NUST Academic Policy QA System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════
# Custom CSS — Premium Dark Theme with Glassmorphism
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 30%, #0d1117 70%, #0a0e27 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(78, 205, 196, 0.3);
        box-shadow: 0 8px 40px rgba(78, 205, 196, 0.1);
        transform: translateY(-2px);
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.08), rgba(69, 183, 209, 0.05));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 16px;
        padding: 28px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(78, 205, 196, 0.08);
    }

    /* Chunk card */
    .chunk-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .chunk-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(255, 255, 255, 0.12);
    }

    /* Title styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ECDC4 0%, #45B7D1 50%, #96CEB4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.5);
        text-align: center;
        margin-bottom: 28px;
        font-weight: 300;
    }

    /* Method badge */
    .method-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-tfidf { background: rgba(255, 107, 107, 0.2); color: #FF6B6B; border: 1px solid rgba(255, 107, 107, 0.3); }
    .badge-minhash { background: rgba(78, 205, 196, 0.2); color: #4ECDC4; border: 1px solid rgba(78, 205, 196, 0.3); }
    .badge-simhash { background: rgba(69, 183, 209, 0.2); color: #45B7D1; border: 1px solid rgba(69, 183, 209, 0.3); }
    .badge-hybrid { background: rgba(150, 206, 180, 0.2); color: #96CEB4; border: 1px solid rgba(150, 206, 180, 0.3); }

    /* Score indicator */
    .score-bar {
        height: 4px;
        border-radius: 2px;
        margin-top: 8px;
        transition: width 0.5s ease;
    }

    /* Metric card */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sample query button */
    .stButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: rgba(78, 205, 196, 0.1) !important;
        border-color: rgba(78, 205, 196, 0.3) !important;
        color: #4ECDC4 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255, 255, 255, 0.02);
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(78, 205, 196, 0.15) !important;
        color: #4ECDC4 !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(78, 205, 196, 0.5) !important;
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.1) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Data Loading & Caching
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_system():
    """Load and index the handbooks."""
    with st.spinner("🔄 Loading handbooks and building indexes..."):
        df = ingest_handbooks()
        preprocess_dataframe(df)

        retriever = HybridRetriever()
        index_timings = retriever.fit(df)

        return df, retriever, index_timings


@st.cache_resource(show_spinner=False)
def load_local_llm():
    """Load the local LLM (cached so it only downloads once)."""
    gen = AnswerGenerator(mode="local_llm")
    gen._init_local_llm()  # Force load now
    return gen


df, retriever, index_timings = load_system()


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    method = st.selectbox(
        "Retrieval Method",
        ["hybrid", "tfidf", "minhash", "simhash"],
        index=0,
        help="Choose the retrieval method for searching",
    )

    top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=5)

    st.markdown("---")
    st.markdown("### 🤖 Answer Generation")

    answer_mode = st.selectbox(
        "Generation Mode",
        ["extractive", "local_llm", "openai"],
        index=0,
        format_func=lambda x: {
            "extractive": "📝 Extractive (no model needed)",
            "local_llm": "🤖 Local LLM (Qwen2.5-3B-Instruct)",
            "openai": "☁️ OpenAI (GPT-4o-mini)",
        }[x],
        help="Choose how answers are generated from retrieved chunks",
    )

    api_key = None
    if answer_mode == "openai":
        api_key = st.text_input("OpenAI API Key", type="password")
    elif answer_mode == "local_llm":
        st.caption("🔧 **Qwen2.5-3B-Instruct** · HuggingFace")
        st.caption("Runs locally on Apple Silicon MPS GPU")
        st.caption("~6GB download on first use")

    st.markdown("---")

    st.markdown("### 📊 System Info")
    stats = retriever.get_stats()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats['num_chunks']}</div>
        <div class="metric-label">Total Chunks</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("TF-IDF Vocab", f"{stats['tfidf_vocab_size']:,}")
        st.metric("MinHash Perms", stats["minhash_num_perm"])
    with col2:
        st.metric("SimHash Bits", stats["simhash_bits"])
        st.metric("LSH Threshold", stats["minhash_threshold"])

    if "pagerank" in stats:
        st.markdown("---")
        st.markdown("### 🔗 PageRank")
        pr = stats["pagerank"]
        st.metric("Sections", pr["num_nodes"])
        st.metric("Cross-references", pr["num_edges"])

    st.markdown("---")
    st.markdown("### ⏱️ Index Build Times")
    for key, val in index_timings.items():
        st.text(f"{key}: {val:.0f} ms")


# ═══════════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-title">🎓 NUST Academic Policy QA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Scalable Retrieval System using LSH, SimHash & TF-IDF</p>', unsafe_allow_html=True)

# Tabs
tab_qa, tab_compare, tab_eval, tab_pagerank = st.tabs([
    "💬 Ask a Question",
    "🔄 Method Comparison",
    "📊 Evaluation Dashboard",
    "🔗 PageRank Analysis",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: Question Answering
# ═══════════════════════════════════════════════════════════════════

with tab_qa:
    # Sample queries
    st.markdown("**Try a sample query:**")
    sample_cols = st.columns(4)
    sample_queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
    ]

    selected_sample = None
    for i, sq in enumerate(sample_queries):
        with sample_cols[i]:
            if st.button(sq, key=f"sample_{i}", use_container_width=True):
                selected_sample = sq

    # Query input
    query = st.text_input(
        "Enter your question about academic policies:",
        value=selected_sample if selected_sample else "",
        placeholder="e.g., What is the minimum GPA requirement for graduation?",
        key="query_input",
    )

    if query:
        # Select the appropriate answer generator based on sidebar mode
        if answer_mode == "local_llm":
            with st.spinner("🤖 Loading local LLM (first time may download ~6GB)..."):
                gen = load_local_llm()
        elif answer_mode == "openai":
            gen = AnswerGenerator(mode="openai", api_key=api_key)
        else:
            gen = AnswerGenerator(mode="extractive")

        with st.spinner("🔍 Retrieving & generating answer..."):
            result = retriever.retrieve(query, method=method, top_k=top_k)
            answer_data = gen.generate(query, result["results"])

        # Method badge
        badge_class = f"badge-{method}" if method != "hybrid" else "badge-hybrid"
        method_label = method.upper()
        if method == "hybrid":
            method_label = "HYBRID (RRF)"

        # Answer display
        st.markdown(f"""
        <div class="answer-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.9rem; color: rgba(255,255,255,0.6); font-weight: 500;">Generated Answer</span>
                <span class="method-badge {badge_class}">{method_label}</span>
            </div>
            <p style="font-size: 1.05rem; line-height: 1.7; color: rgba(255,255,255,0.9); margin: 0;">
                {answer_data['answer']}
            </p>
            <div style="margin-top: 12px; font-size: 0.75rem; color: rgba(255,255,255,0.35);">
                Method: {answer_data['method']} · Latency: {result['timings'].get('total_ms', 0):.1f}ms
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Timing metrics
        st.markdown("#### ⏱️ Performance Metrics")
        timing_cols = st.columns(4)
        timing_labels = ["Total", "TF-IDF", "MinHash LSH", "SimHash"]
        timing_keys = ["total_ms", "tfidf_ms", "minhash_ms", "simhash_ms"]
        for i, (label, key) in enumerate(zip(timing_labels, timing_keys)):
            with timing_cols[i]:
                val = result["timings"].get(key, 0)
                st.metric(label, f"{val:.1f} ms")

        # Retrieved chunks
        st.markdown(f"#### 📄 Top-{top_k} Retrieved Chunks")
        for r in result["results"]:
            source_label = r.get("source", "?").upper()
            pages = f"p.{r.get('page_start', '?')}–{r.get('page_end', '?')}"
            section = r.get("section", "N/A")
            score = r.get("score", 0)
            boost = r.get("pagerank_boost", 1.0)

            # Score color
            score_pct = min(score * 100, 100) if method != "hybrid" else min(score * 1000, 100)
            score_color = "#4ECDC4" if score_pct > 50 else "#45B7D1" if score_pct > 25 else "#FF6B6B"

            contributing = ""
            if "contributing_methods" in r:
                contributing = " · ".join(r["contributing_methods"])

            st.markdown(f"""
            <div class="chunk-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 600; color: #4ECDC4;">#{r.get('rank', '?')}</span>
                        <span style="color: rgba(255,255,255,0.5); margin-left: 8px; font-size: 0.8rem;">
                            {source_label} Handbook · {pages} · {section}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: {score_color}; font-weight: 600;">Score: {score:.4f}</span>
                        {"<br><span style='font-size: 0.7rem; color: rgba(255,255,255,0.4);'>PageRank boost: " + f"{boost:.2f}" + "</span>" if boost != 1.0 else ""}
                        {"<br><span style='font-size: 0.7rem; color: rgba(255,255,255,0.4);'>Methods: " + contributing + "</span>" if contributing else ""}
                    </div>
                </div>
                <p style="margin-top: 12px; color: rgba(255,255,255,0.7); font-size: 0.9rem; line-height: 1.6;">
                    {r.get('text', '')[:500]}{'...' if len(r.get('text', '')) > 500 else ''}
                </p>
                <div class="score-bar" style="width: {score_pct}%; background: linear-gradient(90deg, {score_color}, transparent);"></div>
            </div>
            """, unsafe_allow_html=True)

        # Evidence
        if answer_data.get("evidence"):
            with st.expander("📋 Supporting Evidence (Sentence-level)"):
                for i, ev in enumerate(answer_data["evidence"]):
                    st.markdown(f"""
                    **Evidence {i+1}** (Relevance: {ev.get('relevance_score', 0):.3f})
                    > {ev.get('sentence', '')}

                    *Source: {ev.get('source', 'N/A')} · {ev.get('section', 'N/A')}*
                    """)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: Method Comparison
# ═══════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### 🔄 Side-by-Side Method Comparison")
    st.markdown("Compare retrieval results from all methods for the same query.")

    compare_query = st.text_input(
        "Enter query for comparison:",
        value="What is the minimum GPA requirement?",
        key="compare_query",
    )

    if compare_query:
        methods_list = ["tfidf", "minhash", "simhash", "hybrid"]
        all_results = {}

        for m in methods_list:
            all_results[m] = retriever.retrieve(compare_query, method=m, top_k=5)

        # Latency comparison
        st.markdown("#### ⏱️ Latency Comparison")
        lat_cols = st.columns(4)
        for i, m in enumerate(methods_list):
            with lat_cols[i]:
                lat = all_results[m]["timings"].get("total_ms", 0)
                st.metric(m.upper(), f"{lat:.1f} ms")

        # Results comparison
        cols = st.columns(2)
        for i, m in enumerate(methods_list):
            with cols[i % 2]:
                st.markdown(f"#### {m.upper()}")
                for r in all_results[m]["results"][:3]:
                    st.markdown(f"""
                    **#{r['rank']}** (Score: {r['score']:.4f})
                    *{r.get('source', '?').upper()} · p.{r.get('page_start', '?')} · {r.get('section', 'N/A')}*

                    > {r.get('text', '')[:200]}...
                    """)
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════════
# TAB 3: Evaluation Dashboard
# ═══════════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown("### 📊 Evaluation Dashboard")
    st.markdown("Run comprehensive evaluations comparing all retrieval methods.")

    if st.button("🚀 Run Full Evaluation", key="run_eval", type="primary"):
        with st.spinner("Running evaluations... This may take a few minutes."):
            # Precision/Recall
            st.markdown("#### 1. Precision & Recall")
            eval_df = evaluate_precision_recall(retriever)
            fig_pr = plot_precision_recall_comparison(eval_df)
            st.plotly_chart(fig_pr, use_container_width=True)

            # Show raw data
            with st.expander("📋 Raw Precision/Recall Data"):
                st.dataframe(eval_df.round(3), use_container_width=True)

            # Latency
            st.markdown("#### 2. Query Latency")
            latency_df = evaluate_latency(retriever, num_runs=2)
            fig_lat = plot_latency_comparison(latency_df)
            st.plotly_chart(fig_lat, use_container_width=True)

            # Memory
            st.markdown("#### 3. Memory Usage")
            mem = evaluate_memory(retriever)
            mem_cols = st.columns(3)
            mem_cols[0].metric("RSS Memory", f"{mem['rss_mb']:.0f} MB")
            mem_cols[1].metric("VMS Memory", f"{mem['vms_mb']:.0f} MB")
            mem_cols[2].metric("Chunks Indexed", mem["num_chunks"])

            # Parameter Sensitivity
            st.markdown("#### 4. Parameter Sensitivity")
            mh_df = analyze_minhash_sensitivity(df, queries=GROUND_TRUTH_QUERIES[:3])
            lsh_df = analyze_lsh_bands_sensitivity(df, queries=GROUND_TRUTH_QUERIES[:3])
            sh_df = analyze_simhash_sensitivity(df, queries=GROUND_TRUTH_QUERIES[:3])
            fig_sens = plot_parameter_sensitivity(mh_df, lsh_df, sh_df)
            st.plotly_chart(fig_sens, use_container_width=True)

            # Scalability
            st.markdown("#### 5. Scalability Analysis")
            scale_df = scalability_test(df, scale_factors=[1, 2, 5], queries=GROUND_TRUTH_QUERIES[:2])
            fig_scale = plot_scalability(scale_df)
            st.plotly_chart(fig_scale, use_container_width=True)

    # Load cached results if available
    elif os.path.exists("results/precision_recall.csv"):
        st.markdown("*Showing cached results from last experiment run. Click 'Run Full Evaluation' for fresh data.*")

        eval_df = pd.read_csv("results/precision_recall.csv")
        fig_pr = plot_precision_recall_comparison(eval_df)
        st.plotly_chart(fig_pr, use_container_width=True)

        if os.path.exists("results/latency.csv"):
            latency_df = pd.read_csv("results/latency.csv")
            fig_lat = plot_latency_comparison(latency_df)
            st.plotly_chart(fig_lat, use_container_width=True)

        if os.path.exists("results/scalability.csv"):
            scale_df = pd.read_csv("results/scalability.csv")
            fig_scale = plot_scalability(scale_df)
            st.plotly_chart(fig_scale, use_container_width=True)
    else:
        st.info("Click 'Run Full Evaluation' to generate evaluation metrics and plots.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4: PageRank Analysis
# ═══════════════════════════════════════════════════════════════════

with tab_pagerank:
    st.markdown("### 🔗 PageRank Section Importance")
    st.markdown("Handbook sections ranked by importance using the PageRank algorithm over cross-references.")

    if retriever.pagerank and retriever.pagerank._is_fitted:
        # Graph statistics
        graph_stats = retriever.pagerank.get_graph_stats()
        stat_cols = st.columns(4)
        stat_cols[0].metric("Sections (Nodes)", graph_stats["num_nodes"])
        stat_cols[1].metric("Cross-references (Edges)", graph_stats["num_edges"])
        stat_cols[2].metric("Graph Density", f"{graph_stats['density']:.3f}")
        stat_cols[3].metric("Avg Degree", f"{graph_stats['avg_degree']:.1f}")

        # Top sections
        top_sections = retriever.pagerank.get_top_sections(top_k=15)
        if top_sections:
            st.markdown("#### Top Sections by Importance")

            # Create bar chart
            sec_names = [s[0][:50] for s in top_sections]
            sec_scores = [s[1] for s in top_sections]

            fig = go.Figure(go.Bar(
                y=sec_names[::-1],
                x=sec_scores[::-1],
                orientation="h",
                marker=dict(
                    color=sec_scores[::-1],
                    colorscale=[[0, "#0d1117"], [0.5, "#4ECDC4"], [1, "#45B7D1"]],
                ),
            ))
            fig.update_layout(
                template="plotly_dark",
                height=500,
                margin=dict(l=200, r=20, t=20, b=30),
                xaxis_title="PageRank Score",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table view
            with st.expander("📋 All PageRank Scores"):
                pr_df = pd.DataFrame(top_sections, columns=["Section", "PageRank Score"])
                pr_df["Rank"] = range(1, len(pr_df) + 1)
                st.dataframe(pr_df[["Rank", "Section", "PageRank Score"]].round(6), use_container_width=True)
    else:
        st.warning("PageRank scorer is not available.")
