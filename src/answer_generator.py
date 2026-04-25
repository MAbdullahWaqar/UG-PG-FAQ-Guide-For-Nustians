"""
Answer Generator Module
========================
Generates answers from retrieved chunks using:
1. Local LLM (Qwen2.5-1.5B-Instruct via HuggingFace) — runs on Apple Silicon MPS
2. Groq API (Llama-3.3-70b-versatile) — fast, requires free Groq API key
"""

import os
import gc
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


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an academic policy assistant for NUST (National University of Sciences & Technology). You answer student questions about academic policies based ONLY on the provided handbook excerpts.

IMPORTANT RULES:
1. ONLY use information from the provided context — NEVER make up information.
2. ALWAYS distinguish between UG (Undergraduate/Bachelor's) and PG (Postgraduate/Master's/PhD) policies when BOTH sources are provided.
3. Structure your answer clearly:
   - Start with " For Undergraduate (Bachelor's) Students:" and list the relevant UG policy.
   - Then " For Postgraduate (Master's/PhD) Students:" and list the relevant PG policy.
   - If a policy is the same for both, state that explicitly.
   - If the context only covers one level (UG or PG), answer only for that level.
4. Cite specific sections and page numbers (e.g., "(UG Handbook, p.23)").
5. Be concise but thorough — students need precise numbers, grades, and percentages.
6. Use bullet points for multiple rules or requirements.
7. If the context doesn't contain enough information, say so clearly."""

class AnswerGenerator:
    """
    Generates answers from retrieved chunks.

    Supports two modes:
    - Retrieval Based (Extractive): Extracts and formats the most relevant sentences directly from the handbooks.
    - Groq: Uses Llama-3.1-70b-versatile via Groq API (requires free key)
    """

    def __init__(
        self,
        mode: str = "extractive",
        api_key: str | None = None,
        groq_model: str = "auto",
    ):
        """
        Args:
            mode: One of "extractive", "groq"
            api_key: Groq API key (only for mode="groq")
            groq_model: The specific model to use, or "auto" for fallback cascade
        """
        self.mode = mode
        self.groq_model = groq_model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        # Lazy-loaded resources
        self._groq_client = None
        # Lazy-loaded resources
        self._groq_client = None

        # Initialize based on mode
        if self.mode == "groq":
            self._init_groq()

    def _init_groq(self):
        """Initialize Groq client."""
        if self.api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.api_key)
            except ImportError:
                print("️  Groq not installed. Falling back to extractive mode.")
                self.mode = "extractive"
        else:
            print("️  No Groq API key provided. Falling back to extractive mode.")
            self.mode = "extractive"


    def generate(self, query: str, retrieved_results: list[dict], top_sentences: int = 5) -> dict:
        """
        Generate an answer from retrieved chunks.
        """
        if not retrieved_results:
            return {
                "answer": "I couldn't find relevant information in the handbooks to answer this question.",
                "evidence": [],
                "method": "none",
            }
        if self.mode == "groq" and self.api_key:
            return self._generate_groq(query, retrieved_results)
        else:
            return self._generate_extractive(query, retrieved_results)

    # ------------------------------------------------------------------
    # Extractive Answer Generation
    # ------------------------------------------------------------------

    def _generate_extractive(self, query: str, results: list[dict], top_n: int = 5) -> dict:
        """
        Extract the most relevant sentences from retrieved chunks and format them
        structurally to distinguish between UG and PG, mimicking the Groq mode.
        """
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
                if len(sent) > 20:
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
                "method": "Retrieval Based",
            }

        try:
            vectorizer = TfidfVectorizer()
            corpus = [query] + all_sentences
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            top_indices = sims.argsort()[::-1][:top_n]

            ug_sentences = []
            pg_sentences = []
            evidence = []
            
            for idx in top_indices:
                if sims[idx] > 0.01:
                    sent = all_sentences[idx]
                    src = sentence_sources[idx]["source"]
                    evidence.append({
                        "sentence": sent,
                        "relevance_score": float(sims[idx]),
                        **sentence_sources[idx],
                    })
                    
                    if src == "ug":
                        ug_sentences.append(f"- {sent}")
                    else:
                        pg_sentences.append(f"- {sent}")

            answer_parts = []
            if ug_sentences:
                answer_parts.append("###  For Undergraduate (Bachelor's) Students:\n" + "\n".join(ug_sentences))
            if pg_sentences:
                answer_parts.append("###  For Postgraduate (Master's/PhD) Students:\n" + "\n".join(pg_sentences))
            
            if not answer_parts:
                answer = "No highly relevant specific rules were extracted from the handbooks."
            else:
                answer = "\n\n".join(answer_parts)

        except Exception as e:
            print(f"Exception during TF-IDF calculation: {e}")
            answer = " ".join(all_sentences[:top_n])
            evidence = [{"sentence": s, "relevance_score": 0.0} for s in all_sentences[:top_n]]

        return {
            "answer": answer,
            "evidence": evidence,
            "method": "Retrieval Based",
        }


    def _build_context(self, results: list[dict]) -> tuple[str, list[dict]]:
        """Build context string and evidence list from retrieved results."""
        context_parts = []
        evidence = []
        for i, r in enumerate(results):
            source = r.get("source", "unknown").upper()
            pages = f"p.{r.get('page_start', '?')}-{r.get('page_end', '?')}"
            section = r.get("section", "")
            text = r.get("text", "")
            context_parts.append(
                f"[Source {i+1}: {source} Handbook, {pages}, Section: {section}]\n{text}"
            )
            evidence.append({
                "sentence": text[:200] + "..." if len(text) > 200 else text,
                "relevance_score": float(r.get("score", 0.0)),
                "source": r.get("source", ""),
                "page_start": r.get("page_start", 0),
                "section": section,
            })
        context = "\n\n---\n\n".join(context_parts)
        return context, evidence

    def _build_dual_source_context(self, results: list[dict]) -> tuple[str, list[dict]]:
        """
        Build context organized by source (UG vs PG) for structured answers.
        This helps the LLM distinguish and compare policies across degree levels.
        """
        ug_parts = []
        pg_parts = []
        evidence = []

        for r in results:
            source = r.get("source", "unknown")
            source_label = "UG (Undergraduate)" if source == "ug" else "PG (Postgraduate)"
            pages = f"p.{r.get('page_start', '?')}"
            section = r.get("section", "")
            text = r.get("text", "")

            entry = f"[{source_label} Handbook, {pages}, Section: {section}]\n{text}"

            if source == "ug":
                ug_parts.append(entry)
            else:
                pg_parts.append(entry)

            evidence.append({
                "sentence": text[:200] + "..." if len(text) > 200 else text,
                "relevance_score": float(r.get("score", 0.0)),
                "source": source,
                "page_start": r.get("page_start", 0),
                "section": section,
            })

        context = ""
        if ug_parts:
            context += "=== UG (UNDERGRADUATE / BACHELOR'S) HANDBOOK EXCERPTS ===\n\n"
            context += "\n\n---\n\n".join(ug_parts)
        if pg_parts:
            if context:
                context += "\n\n\n"
            context += "=== PG (POSTGRADUATE / MASTER'S & PhD) HANDBOOK EXCERPTS ===\n\n"
            context += "\n\n---\n\n".join(pg_parts)

        return context, evidence


    def _generate_groq(self, query: str, results: list[dict]) -> dict:
        """
        Generate answer using Groq API with retrieved context.
        Implements an automatic fallback cascade if rate limits are hit.
        """
        context, evidence = self._build_dual_source_context(results)

        user_message = f"""Here are excerpts from NUST's UG (Undergraduate) and PG (Postgraduate) handbooks:

{context}

Student's Question: {query}

Provide a clear, structured answer that distinguishes between UG (Bachelor's) and PG (Master's/PhD) policies where applicable. Use the exact numbers, grades, and percentages from the handbooks:"""

        models_to_try = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768"
        ] if self.groq_model == "auto" else [self.groq_model]

        last_error = None
        for model in models_to_try:
            try:
                response = self._groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=800,
                    temperature=0.2,
                )
                answer = response.choices[0].message.content.strip()
                
                return {
                    "answer": answer,
                    "evidence": evidence,
                    "method": f"Groq API ({model})",
                }
            except Exception as e:
                print(f"️  Groq API error with model {model}: {e}")
                last_error = e

        print(f"️  All Groq models failed. Last error: {last_error}. Falling back to extractive.")
        return self._generate_extractive(query, results)

    @property
    def model_info(self) -> dict:
        """Return information about the current model configuration."""
        info = {"mode": self.mode}
        if self.mode == "extractive":
            info["model_name"] = "Retrieval Based"
        elif self.mode == "groq":
            info["model_name"] = "llama-3.1-8b-instant"
            info["api_key_set"] = bool(self.api_key)
        return info
