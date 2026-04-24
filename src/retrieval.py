"""
Unified Hybrid Retrieval Pipeline
===================================
Combines TF-IDF, MinHash LSH, and SimHash retrievers with
Reciprocal Rank Fusion (RRF) and PageRank boosting.
"""

import time
import psutil
import pandas as pd
from src.preprocessing import preprocess_text, generate_shingles, preprocess_dataframe
from src.tfidf_retriever import TFIDFRetriever
from src.minhash_lsh import MinHashLSHRetriever
from src.simhash_retriever import SimHashRetriever
from src.pagerank import PageRankScorer


class HybridRetriever:
    """
    Unified retrieval system that runs all three methods and
    fuses their results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        # MinHash params
        num_perm: int = 128,
        lsh_threshold: float = 0.05,
        # SimHash params
        hashbits: int = 64,
        hamming_threshold: int = 10,
        # General
        use_pagerank: bool = True,
    ):
        self.tfidf = TFIDFRetriever()
        self.minhash = MinHashLSHRetriever(num_perm=num_perm, threshold=lsh_threshold)
        self.simhash = SimHashRetriever(hashbits=hashbits, hamming_threshold=hamming_threshold)
        self.pagerank = PageRankScorer() if use_pagerank else None

        self.df = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Build all indexes from the chunks DataFrame.
        The DataFrame must have 'processed_text', 'token_set', 'shingles' columns
        (added by preprocessing.preprocess_dataframe).

        Returns timing info for each indexing step.
        """
        self.df = df
        timings = {}

        # 1. TF-IDF Index
        t0 = time.time()
        self.tfidf.fit(
            processed_texts=df["processed_text"].tolist(),
            chunk_ids=df["chunk_id"].tolist(),
        )
        timings["tfidf_index_ms"] = (time.time() - t0) * 1000

        # 2. MinHash LSH Index
        t0 = time.time()
        self.minhash.fit(
            shingle_sets=df["shingles"].tolist(),
            chunk_ids=df["chunk_id"].tolist(),
        )
        timings["minhash_index_ms"] = (time.time() - t0) * 1000

        # 3. SimHash Index
        t0 = time.time()
        # Get TF-IDF weights for SimHash
        tfidf_weights = []
        for text in df["processed_text"]:
            weights = self.tfidf.get_tfidf_weights(text)
            tfidf_weights.append(weights)

        self.simhash.fit(
            token_lists=[text.split() for text in df["processed_text"]],
            chunk_ids=df["chunk_id"].tolist(),
            tfidf_weights=tfidf_weights,
        )
        timings["simhash_index_ms"] = (time.time() - t0) * 1000

        # 4. PageRank
        if self.pagerank:
            t0 = time.time()
            self.pagerank.fit(df)
            timings["pagerank_ms"] = (time.time() - t0) * 1000

        self._is_fitted = True
        return timings

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: Raw user query string
            method: One of 'tfidf', 'minhash', 'simhash', 'hybrid'
            top_k: Number of results to return

        Returns:
            Dict with 'results', 'timings', 'method', and 'memory_mb'
        """
        if not self._is_fitted:
            raise RuntimeError("HybridRetriever has not been fitted.")

        # Preprocess query
        processed_query = preprocess_text(query)
        query_shingles = generate_shingles(processed_query, k=3)
        query_tokens = processed_query.split()

        # Track memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        total_start = time.time()
        timings = {}

        if method == "tfidf":
            results = self.tfidf.retrieve(processed_query, top_k=top_k)
            timings["tfidf_ms"] = results[0]["latency_ms"] if results else 0

        elif method == "minhash":
            results = self.minhash.retrieve(query_shingles, top_k=top_k)
            timings["minhash_ms"] = results[0]["latency_ms"] if results else 0

        elif method == "simhash":
            query_weights = self.tfidf.get_tfidf_weights(processed_query)
            results = self.simhash.retrieve(query_tokens, top_k=top_k, query_weights=query_weights)
            timings["simhash_ms"] = results[0]["latency_ms"] if results else 0

        elif method == "hybrid":
            # Run all three methods
            t0 = time.time()
            tfidf_results = self.tfidf.retrieve(processed_query, top_k=top_k * 2)
            timings["tfidf_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            minhash_results = self.minhash.retrieve(query_shingles, top_k=top_k * 2)
            timings["minhash_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            query_weights = self.tfidf.get_tfidf_weights(processed_query)
            simhash_results = self.simhash.retrieve(query_tokens, top_k=top_k * 2, query_weights=query_weights)
            timings["simhash_ms"] = (time.time() - t0) * 1000

            # Reciprocal Rank Fusion
            results = self._reciprocal_rank_fusion(
                [tfidf_results, minhash_results, simhash_results],
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply PageRank boosting
        if self.pagerank and self.pagerank._is_fitted:
            results = self._apply_pagerank_boost(results)
            # Re-sort after boosting
            results.sort(key=lambda x: x["score"], reverse=True)
            # Re-assign ranks
            for i, r in enumerate(results):
                r["rank"] = i + 1

        total_time = (time.time() - total_start) * 1000
        timings["total_ms"] = total_time

        mem_after = process.memory_info().rss / (1024 * 1024)

        # Enrich results with chunk metadata
        results = self._enrich_results(results)

        return {
            "results": results[:top_k],
            "timings": timings,
            "method": method,
            "memory_mb": mem_after - mem_before,
            "total_memory_mb": mem_after,
            "query": query,
            "processed_query": processed_query,
        }

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict]],
        top_k: int = 5,
        k: int = 60,
    ) -> list[dict]:
        """
        Combine results from multiple methods using Reciprocal Rank Fusion.

        RRF score = sum over methods of 1 / (k + rank)

        Args:
            result_lists: List of result lists from different methods
            top_k: Number of final results
            k: RRF constant (standard value = 60)
        """
        rrf_scores = {}
        chunk_methods = {}

        for results in result_lists:
            for r in results:
                cid = r["chunk_id"]
                rank = r["rank"]
                rrf_score = 1.0 / (k + rank)

                if cid in rrf_scores:
                    rrf_scores[cid] += rrf_score
                else:
                    rrf_scores[cid] = rrf_score

                if cid not in chunk_methods:
                    chunk_methods[cid] = []
                chunk_methods[cid].append(r["method"])

        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (cid, score) in enumerate(sorted_chunks[:top_k]):
            results.append({
                "chunk_id": cid,
                "score": score,
                "rank": rank + 1,
                "method": "Hybrid (RRF)",
                "contributing_methods": list(set(chunk_methods[cid])),
            })

        return results

    def _apply_pagerank_boost(self, results: list[dict]) -> list[dict]:
        """Multiply retrieval scores by PageRank boost factor."""
        if self.df is None:
            return results

        chunk_to_section = dict(zip(self.df["chunk_id"], self.df["section"]))

        for r in results:
            section = chunk_to_section.get(r["chunk_id"], "")
            boost = self.pagerank.get_boost_factor(section)
            r["score"] *= boost
            r["pagerank_boost"] = boost

        return results

    def _enrich_results(self, results: list[dict]) -> list[dict]:
        """Add chunk text, page numbers, and section info to results."""
        if self.df is None:
            return results

        chunk_lookup = self.df.set_index("chunk_id").to_dict("index")

        for r in results:
            cid = r["chunk_id"]
            if cid in chunk_lookup:
                chunk = chunk_lookup[cid]
                r["text"] = chunk.get("text", "")
                r["source"] = chunk.get("source", "")
                r["page_start"] = chunk.get("page_start", 0)
                r["page_end"] = chunk.get("page_end", 0)
                r["section"] = chunk.get("section", "")

        return results

    def retrieve_dual_source(
        self,
        query: str,
        method: str = "hybrid",
        top_k_per_source: int = 3,
    ) -> dict:
        """
        Retrieve chunks from BOTH UG and PG handbooks for comparative answers.

        Ensures representation from each source so the LLM can distinguish
        between undergraduate and postgraduate policies.

        Args:
            query: Raw user query string
            method: Retrieval method to use
            top_k_per_source: Number of top chunks per source (UG/PG)

        Returns:
            Dict with 'ug_results', 'pg_results', 'all_results', 'timings'
        """
        # Retrieve more chunks than needed to ensure both sources are covered
        total_k = top_k_per_source * 4
        full_result = self.retrieve(query, method=method, top_k=total_k)

        # Split by source
        ug_results = [r for r in full_result["results"] if r.get("source") == "ug"]
        pg_results = [r for r in full_result["results"] if r.get("source") == "pg"]

        # Take top-k per source
        ug_results = ug_results[:top_k_per_source]
        pg_results = pg_results[:top_k_per_source]

        # Re-rank within each source
        for i, r in enumerate(ug_results):
            r["rank"] = i + 1
        for i, r in enumerate(pg_results):
            r["rank"] = i + 1

        # Combine for the answer generator
        all_results = ug_results + pg_results

        return {
            "ug_results": ug_results,
            "pg_results": pg_results,
            "all_results": all_results,
            "timings": full_result["timings"],
            "method": full_result["method"],
            "query": query,
            "processed_query": full_result["processed_query"],
        }

    def get_stats(self) -> dict:
        """Return system statistics."""
        stats = {
            "num_chunks": len(self.df) if self.df is not None else 0,
            "tfidf_vocab_size": len(self.tfidf.vectorizer.vocabulary_) if self.tfidf._is_fitted else 0,
            "minhash_num_perm": self.minhash.num_perm,
            "minhash_threshold": self.minhash.threshold,
            "simhash_bits": self.simhash.hashbits,
            "simhash_hamming_threshold": self.simhash.hamming_threshold,
        }
        if self.pagerank and self.pagerank._is_fitted:
            stats["pagerank"] = self.pagerank.get_graph_stats()
        return stats

