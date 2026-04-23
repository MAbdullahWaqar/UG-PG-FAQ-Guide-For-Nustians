"""
SimHash Retriever
==================
Custom implementation of SimHash using weighted token hashing
and Hamming distance for approximate similarity detection.

No external SimHash library is used — this is fully hand-crafted.
"""

import time
import hashlib
import numpy as np
from collections import Counter


class SimHashRetriever:
    """
    Approximate retrieval using SimHash fingerprints and Hamming distance.

    Algorithm:
    1. For each chunk, compute a binary fingerprint by:
       a. Hashing each token/shingle to a fixed-length binary string (MD5)
       b. Weighting each hash by TF-IDF score (or term frequency)
       c. Aggregating into a single fingerprint via weighted voting
    2. For queries, compute the same fingerprint
    3. Find chunks with minimum Hamming distance to query

    This is a custom implementation — no SimHash library is used.
    """

    def __init__(self, hashbits: int = 64, hamming_threshold: int = 10):
        """
        Args:
            hashbits: Length of the SimHash fingerprint in bits
            hamming_threshold: Maximum Hamming distance to consider similar
        """
        self.hashbits = hashbits
        self.hamming_threshold = hamming_threshold
        self.fingerprints = {}  # chunk_id -> int (fingerprint)
        self.chunk_ids = []
        self._is_fitted = False

    def _token_hash(self, token: str) -> list[int]:
        """
        Hash a token to a list of +1/-1 values using MD5.
        Each bit of the MD5 hash maps to +1 (bit=1) or -1 (bit=0).
        We only use the first `hashbits` bits.
        """
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        # Convert hex to binary
        binary = bin(int(h, 16))[2:].zfill(128)
        # Take first hashbits bits, convert to +1/-1
        return [1 if binary[i] == "1" else -1 for i in range(self.hashbits)]

    def _compute_fingerprint(self, tokens: list[str], weights: dict[str, float] | None = None) -> int:
        """
        Compute the SimHash fingerprint for a list of tokens.

        Args:
            tokens: List of tokens/shingles
            weights: Optional dict of token -> weight (e.g., TF-IDF scores)
                    If None, uses term frequency as weight.
        """
        if not tokens:
            return 0

        # Initialize vote vector
        v = np.zeros(self.hashbits, dtype=np.float64)

        # Count token frequencies as default weights
        if weights is None:
            freq = Counter(tokens)
            weights = {t: float(c) for t, c in freq.items()}

        for token in set(tokens):
            w = weights.get(token, 1.0)
            hash_bits = self._token_hash(token)
            for i in range(self.hashbits):
                v[i] += hash_bits[i] * w

        # Convert to binary fingerprint
        fingerprint = 0
        for i in range(self.hashbits):
            if v[i] > 0:
                fingerprint |= (1 << (self.hashbits - 1 - i))

        return fingerprint

    def _hamming_distance(self, fp1: int, fp2: int) -> int:
        """Compute Hamming distance between two fingerprints."""
        xor = fp1 ^ fp2
        return bin(xor).count("1")

    def fit(
        self,
        token_lists: list[list[str]],
        chunk_ids: list[str],
        tfidf_weights: list[dict[str, float]] | None = None,
    ) -> None:
        """
        Build the SimHash index.

        Args:
            token_lists: List of token lists for each chunk
            chunk_ids: Corresponding chunk identifiers
            tfidf_weights: Optional TF-IDF weights per chunk for better fingerprints
        """
        self.chunk_ids = list(chunk_ids)
        self.fingerprints = {}

        for i, (cid, tokens) in enumerate(zip(chunk_ids, token_lists)):
            w = tfidf_weights[i] if tfidf_weights else None
            fp = self._compute_fingerprint(tokens, weights=w)
            self.fingerprints[cid] = fp

        self._is_fitted = True

    def retrieve(
        self,
        query_tokens: list[str],
        top_k: int = 5,
        query_weights: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks with minimum Hamming distance to query.

        Args:
            query_tokens: List of tokens from the query
            top_k: Number of results to return
            query_weights: Optional TF-IDF weights for query tokens
        """
        if not self._is_fitted:
            raise RuntimeError("SimHashRetriever has not been fitted.")

        start_time = time.time()

        query_fp = self._compute_fingerprint(query_tokens, weights=query_weights)

        # Compute Hamming distance to all chunks
        distances = []
        for cid, fp in self.fingerprints.items():
            dist = self._hamming_distance(query_fp, fp)
            distances.append((cid, dist))

        # Sort by distance (ascending — lower is more similar)
        distances.sort(key=lambda x: x[1])

        elapsed = time.time() - start_time

        results = []
        for rank, (cid, dist) in enumerate(distances[:top_k]):
            # Convert Hamming distance to a similarity score (0 to 1)
            similarity = 1.0 - (dist / self.hashbits)
            results.append({
                "chunk_id": cid,
                "score": float(similarity),
                "hamming_distance": dist,
                "rank": rank + 1,
                "method": "SimHash",
                "latency_ms": elapsed * 1000,
            })

        return results

    def get_within_threshold(self, query_tokens: list[str]) -> int:
        """Return count of chunks within the Hamming threshold (for analysis)."""
        query_fp = self._compute_fingerprint(query_tokens)
        count = 0
        for fp in self.fingerprints.values():
            if self._hamming_distance(query_fp, fp) <= self.hamming_threshold:
                count += 1
        return count
