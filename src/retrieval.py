"""
Retrieval-for-grounding: given a new customer message, find the most similar
historical (customer message -> brand reply) pairs from the same brand so the
reply drafter can imitate how the brand *actually* resolved similar issues,
instead of hallucinating a generic support-bot response.

Design choice: BM25 over raw text, not embeddings. At this scale (a few
thousand historical pairs per brand) BM25 is fast, needs no extra API calls
or vector DB, and is easy to debug/explain live -- which matters given the
"explain your own code" interview format. See DECISION_LOG.md.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

try:
    from rank_bm25 import BM25Okapi
    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover - exercised only in offline/dev envs
    _HAS_RANK_BM25 = False


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class _MiniBM25:
    """Minimal BM25 implementation with the same .get_scores() interface as
    rank_bm25.BM25Okapi. Used only as a fallback when rank_bm25 isn't
    installed (e.g. offline dev environments) -- functionally equivalent
    for our purposes, just without rank_bm25's micro-optimizations.
    Reference: Robertson & Zaragoza, "The Probabilistic Relevance
    Framework: BM25 and Beyond" (2009), standard BM25 formula, k1=1.5, b=0.75.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / max(len(corpus), 1)
        self.df = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1
        self.n_docs = len(corpus)
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            dl = self.doc_len[i]
            for term in query:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                numer = tf[term] * (self.k1 + 1)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * numer / denom
        return scores


@dataclass
class HistoricalCase:
    customer_text: str
    brand_reply: str
    intent: str | None = None


class ResolutionIndex:
    """BM25 index over historical resolved (customer_text -> brand_reply) pairs."""

    def __init__(self, cases: list[HistoricalCase]):
        self.cases = cases
        self._corpus_tokens = [_tokenize(c.customer_text) for c in cases]
        if _HAS_RANK_BM25:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = _MiniBM25(self._corpus_tokens)

    @classmethod
    def from_dataframe(cls, df, text_col="customer_text_clean",
                        reply_col="brand_reply_text_clean", intent_col=None):
        cases = []
        for _, row in df.iterrows():
            if not isinstance(row.get(reply_col), str) or not row.get(reply_col):
                continue
            cases.append(HistoricalCase(
                customer_text=row[text_col],
                brand_reply=row[reply_col],
                intent=row.get(intent_col) if intent_col else None,
            ))
        return cls(cases)

    def search(self, query: str, k: int = 3, intent_filter: str | None = None) -> list[HistoricalCase]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.cases)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            case = self.cases[i]
            if intent_filter and case.intent and case.intent != intent_filter:
                continue
            if scores[i] <= 0:
                continue
            results.append(case)
            if len(results) >= k:
                break
        return results
