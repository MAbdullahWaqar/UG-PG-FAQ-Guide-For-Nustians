"""
Answer Generator Module
========================
Generates answers from retrieved chunks using:
1. Extractive method (sentence-level TF-IDF matching) — default
2. LLM-based generation (OpenAI API) — optional
"""

import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class AnswerGenerator:
    """
    Generates answers from retrieved chunks.

    Supports two modes:
    - Extractive: Selects most relevant sentences from chunks
    - LLM: Uses OpenAI API with retrieved context (if API key provided)
    """

    def __init__(self, use_llm: bool = False, api_key: str | None = None):
        self.use_llm = use_llm
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

        if self.use_llm and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("⚠️  OpenAI not installed. Falling back to extractive mode.")
                self.use_llm = False

    def generate(self, query: str, retrieved_results: list[dict], top_sentences: int = 5) -> dict:
        """
        Generate an answer from retrieved chunks.

        Args:
            query: Original user question
            retrieved_results: List of result dicts with 'text' field
            top_sentences: Number of sentences for extractive answer

        Returns:
            Dict with 'answer', 'evidence', 'method'
        """
        if not retrieved_results:
            return {
                "answer": "I couldn't find relevant information in the handbooks to answer this question.",
                "evidence": [],
                "method": "none",
            }

        if self.use_llm and self._client:
            return self._generate_llm(query, retrieved_results)
        else:
            return self._generate_extractive(query, retrieved_results, top_sentences)

    def _generate_extractive(self, query: str, results: list[dict], top_n: int = 5) -> dict:
        """
        Extract the most relevant sentences from retrieved chunks.

        Uses sentence-level TF-IDF similarity to rank sentences.
        """
        # Collect all sentences from retrieved chunks
        all_sentences = []
        sentence_sources = []

        for r in results:
            text = r.get("text", "")
            if not text:
                continue
            try:
                sentences = sent_tokenize(text)
            except Exception:
                sentences = text.split(". ")
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:  # Skip very short fragments
                    all_sentences.append(sent)
                    sentence_sources.append({
                        "chunk_id": r.get("chunk_id", ""),
                        "source": r.get("source", ""),
                        "page_start": r.get("page_start", 0),
                        "section": r.get("section", ""),
                    })

        if not all_sentences:
            return {
                "answer": "Retrieved chunks did not contain enough text to generate an answer.",
                "evidence": [],
                "method": "extractive",
            }

        # Rank sentences by TF-IDF similarity to query
        try:
            vectorizer = TfidfVectorizer()
            corpus = [query] + all_sentences
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Get top-N sentence indices
            top_indices = sims.argsort()[::-1][:top_n]

            # Build answer
            answer_sentences = []
            evidence = []
            for idx in top_indices:
                if sims[idx] > 0.01:  # Only include somewhat relevant sentences
                    answer_sentences.append(all_sentences[idx])
                    evidence.append({
                        "sentence": all_sentences[idx],
                        "relevance_score": float(sims[idx]),
                        **sentence_sources[idx],
                    })

            answer = " ".join(answer_sentences) if answer_sentences else all_sentences[0]

        except Exception:
            # Fallback: just return first few sentences
            answer = " ".join(all_sentences[:top_n])
            evidence = [{"sentence": s, "relevance_score": 0.0} for s in all_sentences[:top_n]]

        return {
            "answer": answer,
            "evidence": evidence,
            "method": "extractive",
        }

    def _generate_llm(self, query: str, results: list[dict]) -> dict:
        """
        Generate answer using OpenAI API with retrieved context.
        The answer must be grounded in the retrieved content.
        """
        # Build context from retrieved chunks
        context_parts = []
        evidence = []
        for i, r in enumerate(results):
            source = r.get("source", "unknown").upper()
            pages = f"p.{r.get('page_start', '?')}-{r.get('page_end', '?')}"
            section = r.get("section", "")
            text = r.get("text", "")
            context_parts.append(f"[Source {i+1}: {source} Handbook, {pages}, Section: {section}]\n{text}")
            evidence.append({
                "sentence": text[:200] + "..." if len(text) > 200 else text,
                "source": r.get("source", ""),
                "page_start": r.get("page_start", 0),
                "section": section,
            })

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are an academic policy assistant for NUST university. Answer the student's question based ONLY on the provided handbook excerpts. 

Rules:
1. Only use information from the provided context
2. Cite specific sections and page numbers when possible
3. If the context doesn't contain enough information, say so
4. Be concise but thorough

Context from handbooks:
{context}

Student's Question: {query}

Answer:"""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to extractive if API fails
            print(f"⚠️  LLM API error: {e}. Falling back to extractive.")
            return self._generate_extractive(query, results)

        return {
            "answer": answer,
            "evidence": evidence,
            "method": "llm (gpt-4o-mini)",
        }
