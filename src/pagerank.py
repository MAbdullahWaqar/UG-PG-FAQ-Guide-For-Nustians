"""
PageRank Section Importance Scorer
===================================
Extension component: Uses PageRank to rank important sections
of the handbooks based on cross-references and structural links.
"""

import re
import networkx as nx
import pandas as pd


class PageRankScorer:
    """
    Builds a directed graph of handbook sections and computes
    PageRank scores to identify the most important sections.

    Edges are created from:
    1. Cross-references (e.g., "See Section 3.2", "as per rule 5.1")
    2. Sequential section flow (natural reading order)
    3. Shared terminology (sections that discuss similar topics)
    """

    def __init__(self, damping: float = 0.85):
        """
        Args:
            damping: PageRank damping factor (probability of following a link)
        """
        self.damping = damping
        self.graph = nx.DiGraph()
        self.scores = {}
        self._is_fitted = False

    def _extract_references(self, text: str) -> list[str]:
        """
        Extract cross-references from text.
        Looks for patterns like "Section 3.2", "Rule 5", "Clause 7.1", etc.
        """
        patterns = [
            r"(?:section|sec\.|rule|clause|article|para|paragraph)\s*(\d+\.?\d*\.?\d*)",
            r"(?:refer|see|as per|according to|mentioned in)\s+(?:section|sec\.|rule|clause)?\s*(\d+\.?\d*\.?\d*)",
        ]
        refs = []
        text_lower = text.lower()
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            refs.extend(matches)
        return refs

    def _extract_section_number(self, section_title: str) -> str | None:
        """Extract the numeric part of a section title."""
        match = re.match(r"(\d+\.?\d*\.?\d*)", section_title.strip())
        return match.group(1) if match else None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Build the section graph and compute PageRank.

        Args:
            df: DataFrame with 'chunk_id', 'section', 'text' columns
        """
        self.graph = nx.DiGraph()

        # Get unique sections
        sections = df["section"].unique().tolist()
        section_nums = {}
        for sec in sections:
            num = self._extract_section_number(sec)
            if num:
                section_nums[num] = sec
            self.graph.add_node(sec)

        # Add edges from cross-references
        for _, row in df.iterrows():
            source_section = row["section"]
            refs = self._extract_references(row["text"])
            for ref_num in refs:
                # Find the best matching section
                if ref_num in section_nums:
                    target_section = section_nums[ref_num]
                    if target_section != source_section:
                        if self.graph.has_edge(source_section, target_section):
                            self.graph[source_section][target_section]["weight"] += 1
                        else:
                            self.graph.add_edge(source_section, target_section, weight=1)
                else:
                    # Try prefix matching (e.g., ref "3" matches section "3.1")
                    for num, sec in section_nums.items():
                        if num.startswith(ref_num) or ref_num.startswith(num):
                            if sec != source_section:
                                if self.graph.has_edge(source_section, sec):
                                    self.graph[source_section][sec]["weight"] += 1
                                else:
                                    self.graph.add_edge(source_section, sec, weight=1)

        # Add sequential edges (reading order)
        for i in range(len(sections) - 1):
            self.graph.add_edge(sections[i], sections[i + 1], weight=0.5)

        # Compute PageRank
        if len(self.graph.nodes()) > 0:
            try:
                self.scores = nx.pagerank(
                    self.graph,
                    alpha=self.damping,
                    weight="weight",
                    max_iter=100,
                )
            except nx.NetworkXError:
                # Fallback: uniform scores
                n = len(self.graph.nodes())
                self.scores = {node: 1.0 / n for node in self.graph.nodes()}
        else:
            self.scores = {}

        self._is_fitted = True

    def get_score(self, section: str) -> float:
        """Get the PageRank score for a section."""
        return self.scores.get(section, 0.0)

    def get_boost_factor(self, section: str, max_boost: float = 1.5) -> float:
        """
        Get a multiplicative boost factor for retrieval results.
        Maps PageRank score to [1.0, max_boost] range.
        """
        if not self.scores:
            return 1.0

        score = self.get_score(section)
        if score == 0:
            return 1.0

        max_score = max(self.scores.values()) if self.scores else 1.0
        min_score = min(self.scores.values()) if self.scores else 0.0

        if max_score == min_score:
            return 1.0

        # Normalize to [0, 1] then map to [1.0, max_boost]
        normalized = (score - min_score) / (max_score - min_score)
        return 1.0 + normalized * (max_boost - 1.0)

    def get_top_sections(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k most important sections."""
        sorted_sections = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_sections[:top_k]

    def get_graph_stats(self) -> dict:
        """Return statistics about the section graph."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
        }
