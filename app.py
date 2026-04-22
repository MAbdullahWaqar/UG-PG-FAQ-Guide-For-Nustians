import streamlit as st
import pandas as pd
import os
from src.ingestion import ingest_handbooks
from src.preprocessing import preprocess_chunks
from src.retrieval import QA_Retriever
from src.answer_generation import ExtractiveAnswerGenerator

# Page config
st.set_page_config(page_title="Academic Policy QA", page_icon="🎓", layout="wide")

# Initialize models
@st.cache_resource
def load_system():
    # Check if processed chunks exist
    if os.path.exists("processed_chunks.csv"):
        df = pd.read_csv("processed_chunks.csv")
    else:
        with st.spinner("Ingesting and processing PDFs for the first time. This may take a minute..."):
            df_raw = ingest_handbooks(chunk_size=200)
            if df_raw.empty:
                st.error("No PDFs found in the 'Handbooks' directory.")
                st.stop()
            df = preprocess_chunks(df_raw)
            df.to_csv("processed_chunks.csv", index=False)
            
    # Initialize and fit retriever
    retriever = QA_Retriever()
    with st.spinner("Building indices (TF-IDF, MinHash+LSH, SimHash)..."):
        retriever.fit_all(df)
        
    # Initialize generator
    generator = ExtractiveAnswerGenerator()
    
    return retriever, generator

st.title("🎓 Scalable Academic Policy QA System")
st.markdown("Ask questions about UG and PG handbooks. Powered by Big Data Retrieval Techniques.")

try:
    retriever, generator = load_system()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# Sidebar for settings
st.sidebar.header("Settings")
retrieval_method = st.sidebar.radio(
    "Retrieval Method",
    options=["TF-IDF", "MinHash_LSH", "SimHash"],
    index=0,
    help="TF-IDF is exact. MinHash_LSH and SimHash are approximate."
)
top_k = st.sidebar.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=3)
use_reranking = st.sidebar.checkbox("Use Position Re-ranking (Extension)", value=False, help="Boost score of chunks appearing earlier in documents.")

# Main search interface
query = st.text_input("Enter your question:", placeholder="e.g., What is the minimum GPA requirement?")

if query:
    with st.spinner(f"Retrieving using {retrieval_method}..."):
        if use_reranking:
            results = retriever.retrieve_with_reranking(query, method=retrieval_method, top_k=top_k)
        else:
            results = retriever.retrieve(query, method=retrieval_method, top_k=top_k)
            
        if not results:
            st.warning("No relevant information found. Try rephrasing your query or using TF-IDF.")
        else:
            # Generate Answer
            answer = generator.generate(query, results)
            
            # Display Final Answer
            st.subheader("💡 Answer")
            st.success(answer)
            
            # Display Retrieved Chunks
            st.subheader(f"📚 Retrieved Context ({retrieval_method})")
            
            for i, res in enumerate(results):
                with st.expander(f"Source {i+1}: {res['document'].upper()} Handbook, Page {res['page_number']}", expanded=True if i==0 else False):
                    st.markdown(f"**Chunk ID:** `{res['chunk_id']}` | **Similarity Score:** `{res['similarity_score']:.4f}`")
                    st.write(res['text'])
