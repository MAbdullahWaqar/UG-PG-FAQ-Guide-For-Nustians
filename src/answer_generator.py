"""
Answer Generator Module
========================
Generates answers from retrieved chunks using:
1. Extractive method (sentence-level TF-IDF matching) — default
2. Local LLM (Qwen2.5-3B-Instruct via HuggingFace) — runs on Apple Silicon MPS
3. OpenAI API (GPT-4o-mini) — optional, requires API key
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
# System prompt for grounded academic QA
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an academic policy assistant for NUST (National University of Sciences & Technology). You answer student questions about academic policies based ONLY on the provided handbook excerpts.

IMPORTANT RULES:
1. ONLY use information from the provided context — NEVER make up information.
2. ALWAYS distinguish between UG (Undergraduate/Bachelor's) and PG (Postgraduate/Master's/PhD) policies when BOTH sources are provided.
3. Structure your answer clearly:
   - Start with "📘 For Undergraduate (Bachelor's) Students:" and list the relevant UG policy.
   - Then "📗 For Postgraduate (Master's/PhD) Students:" and list the relevant PG policy.
   - If a policy is the same for both, state that explicitly.
   - If the context only covers one level (UG or PG), answer only for that level.
4. Cite specific sections and page numbers (e.g., "(UG Handbook, p.23)").
5. Be concise but thorough — students need precise numbers, grades, and percentages.
6. Use bullet points for multiple rules or requirements.
7. If the context doesn't contain enough information, say so clearly."""


class AnswerGenerator:
    """
    Generates answers from retrieved chunks.

    Supports three modes:
    - Extractive: Selects most relevant sentences from chunks (default, no model needed)
    - Local LLM: Uses Qwen2.5-3B-Instruct running locally on Apple Silicon MPS
    - OpenAI: Uses GPT-4o-mini via API (requires key)
    """

    def __init__(
        self,
        mode: str = "extractive",
        api_key: str | None = None,
        local_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    ):
        """
        Args:
            mode: One of "extractive", "local_llm", "openai"
            api_key: OpenAI API key (only for mode="openai")
            local_model_name: HuggingFace model ID for local LLM
        """
        self.mode = mode
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.local_model_name = local_model_name

        # Lazy-loaded resources
        self._openai_client = None
        self._local_model = None
        self._local_tokenizer = None
        self._device = None

        # Initialize based on mode
        if self.mode == "openai":
            self._init_openai()
        elif self.mode == "local_llm":
            # Model is loaded lazily on first generate() call to avoid
            # blocking the UI startup
            pass

    def _init_openai(self):
        """Initialize OpenAI client."""
        if self.api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("⚠️  OpenAI not installed. Falling back to extractive mode.")
                self.mode = "extractive"
        else:
            print("⚠️  No OpenAI API key provided. Falling back to extractive mode.")
            self.mode = "extractive"

    def _init_local_llm(self):
        """
        Initialize the local Qwen2.5-3B-Instruct model.
        Uses Apple Silicon MPS for GPU acceleration.
        """
        if self._local_model is not None:
            return  # Already loaded

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"🤖 Loading local LLM: {self.local_model_name}")
        print("   This may take a minute on first download (~6GB)...")

        # Determine device
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            print("   → Using Apple Silicon MPS GPU acceleration")
        else:
            self._device = torch.device("cpu")
            print("   → Using CPU (MPS not available)")

        # Load tokenizer
        self._local_tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_name,
            trust_remote_code=True,
        )

        # Load model in float16 for efficiency on M-series chips
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.local_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"   ✅ Model loaded successfully on {self._device}")

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

        if self.mode == "local_llm":
            return self._generate_local_llm(query, retrieved_results)
        elif self.mode == "openai" and self._openai_client:
            return self._generate_openai(query, retrieved_results)
        else:
            return self._generate_extractive(query, retrieved_results, top_sentences)

    # ------------------------------------------------------------------
    # Extractive Answer Generation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Local LLM Answer Generation (Qwen2.5-3B-Instruct)
    # ------------------------------------------------------------------

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

    def _generate_local_llm(self, query: str, results: list[dict]) -> dict:
        """
        Generate answer using local Qwen2.5-3B-Instruct model.
        Runs on Apple Silicon MPS for fast inference.
        Uses dual-source context to give structured UG vs PG answers.
        """
        import torch

        # Lazy load model on first call
        self._init_local_llm()

        # Use dual-source context for structured UG vs PG answers
        context, evidence = self._build_dual_source_context(results)

        user_message = f"""Here are excerpts from NUST's UG (Undergraduate) and PG (Postgraduate) handbooks:

{context}

Student's Question: {query}

Provide a clear, structured answer that distinguishes between UG (Bachelor's) and PG (Master's/PhD) policies where applicable. Use the exact numbers, grades, and percentages from the handbooks:"""

        # Build chat messages in Qwen format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            # Apply chat template
            text = self._local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = self._local_tokenizer(
                [text], return_tensors="pt"
            ).to(self._local_model.device)

            # Generate with conservative settings for factual accuracy
            with torch.no_grad():
                generated_ids = self._local_model.generate(
                    **model_inputs,
                    max_new_tokens=400,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                )

            # Decode — only the new tokens (skip input)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            answer = self._local_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            # Clean up GPU memory
            del model_inputs, generated_ids, output_ids
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"⚠️  Local LLM error: {e}. Falling back to extractive.")
            return self._generate_extractive(query, results)

        return {
            "answer": answer,
            "evidence": evidence,
            "method": f"local LLM ({self.local_model_name.split('/')[-1]})",
        }

    # ------------------------------------------------------------------
    # OpenAI API Answer Generation
    # ------------------------------------------------------------------

    def _generate_openai(self, query: str, results: list[dict]) -> dict:
        """
        Generate answer using OpenAI API with retrieved context.
        The answer must be grounded in the retrieved content.
        """
        context, evidence = self._build_dual_source_context(results)

        user_message = f"""Here are excerpts from NUST's UG (Undergraduate) and PG (Postgraduate) handbooks:

{context}

Student's Question: {query}

Provide a clear, structured answer that distinguishes between UG (Bachelor's) and PG (Master's/PhD) policies where applicable. Use the exact numbers, grades, and percentages from the handbooks:"""

        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=500,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to extractive if API fails
            print(f"⚠️  OpenAI API error: {e}. Falling back to extractive.")
            return self._generate_extractive(query, results)

        return {
            "answer": answer,
            "evidence": evidence,
            "method": "LLM (GPT-4o-mini)",
        }

    # ------------------------------------------------------------------
    # Resource Management
    # ------------------------------------------------------------------

    def unload_model(self):
        """Unload the local LLM from memory to free GPU/RAM."""
        if self._local_model is not None:
            import torch
            del self._local_model
            del self._local_tokenizer
            self._local_model = None
            self._local_tokenizer = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print("🗑️  Local LLM unloaded from memory.")

    @property
    def is_model_loaded(self) -> bool:
        """Check if the local LLM is currently loaded."""
        return self._local_model is not None

    @property
    def model_info(self) -> dict:
        """Return information about the current model configuration."""
        info = {"mode": self.mode}
        if self.mode == "local_llm":
            info["model_name"] = self.local_model_name
            info["loaded"] = self.is_model_loaded
            if self._device:
                info["device"] = str(self._device)
        elif self.mode == "openai":
            info["model_name"] = "gpt-4o-mini"
            info["api_key_set"] = bool(self.api_key)
        return info
