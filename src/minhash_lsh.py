"""
MinHash + LSH Retriever
========================
Approximate retrieval using MinHash signatures and Locality Sensitive Hashing.
Uses shingle-based set representation for better accuracy with short queries.
"""

import time
import numpy as np
from datasketch import MinHash, MinHashLSH


class MinHashLSHRetriever:
    """
    Approximate retrieval using MinHash + LSH.

    Key design decisions:
    - Uses word shingles (not raw tokens) for better Jaccard estimation
    - Low threshold for asymmetric short-query-to-long-document matching
    - Re-ranks LSH candidates using actual MinHash Jaccard estimates
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.05):
        """
        Args:
            num_perm: Number of permutation functions for MinHash
            threshold: Jaccard similarity threshold for LSH
                       (lower = more candidates = higher recall)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.signatures = {}  # chunk_id -> MinHash
        self.chunk_ids = []
        self._is_fitted = False

    def _create_minhash(self, token_set: set) -> MinHash:
        """Create a MinHash signature from a set of tokens/shingles."""
        m = MinHash(num_perm=self.num_perm)
        for item in token_set:
            m.update(item.encode("utf-8"))
        return m

    def fit(self, shingle_sets: list[set], chunk_ids: list[str]) -> None:
        """
        Build the MinHash LSH index.

        Args:
            shingle_sets: List of shingle sets for each chunk
            chunk_ids: Corresponding chunk identifiers
        """
        self.chunk_ids = list(chunk_ids)
        # Recreate LSH to avoid stale state
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.signatures = {}

        for cid, shingles in zip(chunk_ids, shingle_sets):
            if not shingles:
                continue
            mh = self._create_minhash(shingles)
            self.signatures[cid] = mh
            try:
                self.lsh.insert(cid, mh)
            except ValueError:
                # Duplicate key, skip
                pass

        self._is_fitted = True

    def retrieve(self, query_shingles: set, top_k: int = 5) -> list[dict]:
        """
        Retrieve top-k similar chunks using MinHash LSH.

        Steps:
        1. Create MinHash for query
        2. Query LSH for candidate set (approximate)
        3. Re-rank candidates by actual MinHash Jaccard estimate

        Args:
            query_shingles: Set of shingles from the query
            top_k: Number of results to return
        """
        if not self._is_fitted:
            raise RuntimeError("MinHashLSHRetriever has not been fitted.")

        start_time = time.time()

        # Create query MinHash
        query_mh = self._create_minhash(query_shingles)

        # Get LSH candidates
        candidates = self.lsh.query(query_mh)

        # If LSH returns too few candidates, do brute-force on a sample
        if len(candidates) < top_k:
            # Fallback: compute Jaccard against all chunks
            all_scores = []
            for cid, mh in self.signatures.items():
                score = query_mh.jaccard(mh)
                all_scores.append((cid, score))
            all_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [cid for cid, _ in all_scores[:top_k * 3]]

        # Re-rank candidates by actual Jaccard estimate
        scored = []
        for cid in candidates:
            if cid in self.signatures:
                jaccard = query_mh.jaccard(self.signatures[cid])
                scored.append((cid, jaccard))

        scored.sort(key=lambda x: x[1], reverse=True)

        elapsed = time.time() - start_time

        results = []
        for rank, (cid, score) in enumerate(scored[:top_k]):
            results.append({
                "chunk_id": cid,
                "score": float(score),
                "rank": rank + 1,
                "method": "MinHash-LSH",
                "latency_ms": elapsed * 1000,
            })

        return results

    def get_num_candidates(self, query_shingles: set) -> int:
        """Return the number of LSH candidates (for analysis)."""
        query_mh = self._create_minhash(query_shingles)
        return len(self.lsh.query(query_mh))
