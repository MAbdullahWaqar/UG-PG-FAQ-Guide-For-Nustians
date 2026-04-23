"""
TF-IDF Baseline Retriever
==========================
Exact retrieval method using TF-IDF vectorization and cosine similarity.
This serves as the baseline for comparison against approximate methods.
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    """
    Exact retrieval using TF-IDF + Cosine Similarity.

    This is the baseline (non-approximate) method required for
    comparison against LSH-based approximate methods.
    """

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        """
        Args:
            max_features: Maximum vocabulary size
            ngram_range: Use unigrams and bigrams for better matching
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,  # Apply sublinear TF scaling (1 + log(tf))
        )
        self.tfidf_matrix = None
        self.chunk_ids = None
        self._is_fitted = False

    def fit(self, processed_texts: list[str], chunk_ids: list[str]) -> None:
        """
        Build the TF-IDF index over all chunks.

        Args:
            processed_texts: List of preprocessed text strings
            chunk_ids: Corresponding chunk identifiers
        """
        self.chunk_ids = list(chunk_ids)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        self._is_fitted = True

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top-k most similar chunks to the query.

        Args:
            query: Preprocessed query text
            top_k: Number of results to return

        Returns:
            List of dicts with 'chunk_id', 'score', 'rank'
        """
        if not self._is_fitted:
            raise RuntimeError("TFIDFRetriever has not been fitted. Call fit() first.")

        start_time = time.time()

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        elapsed = time.time() - start_time

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": float(similarities[idx]),
                "rank": rank + 1,
                "method": "TF-IDF",
                "latency_ms": elapsed * 1000,
            })

        return results

    def get_tfidf_weights(self, text: str) -> dict[str, float]:
        """
        Get TF-IDF weights for terms in a text.
        Useful for SimHash weighting.
        """
        if not self._is_fitted:
            return {}
        vec = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        weights = {}
        for idx in vec.nonzero()[1]:
            weights[feature_names[idx]] = float(vec[0, idx])
        return weights
