"""Retrieval metrics for melodic pattern recognition.

Every phrase in the test set is used in turn as a query against all the others, and
a retrieval is relevant when it carries the same phrase identifier. The three
metrics answer three different questions: how good the ranking is overall (MAP),
how quickly a first correct phrase appears (MRR), and how clean the very top of the
list is (P@k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalPrecision

#: Cut-offs reported for Precision@k.
PRECISION_CUTOFFS = (1, 5)


@dataclass(frozen=True)
class RetrievalScores:
    """Mean Average Precision, Mean Reciprocal Rank and Precision at each cut-off."""

    mean_average_precision: float
    mean_reciprocal_rank: float
    precision_at_k: dict[int, float]

    def as_row(self) -> dict[str, float]:
        """Flat, machine-readable form. Keys avoid ``@`` so that they are valid
        MLflow metric names as well as column headers."""
        return {
            "map": self.mean_average_precision,
            "mrr": self.mean_reciprocal_rank,
            **{f"precision_at_{k}": value for k, value in self.precision_at_k.items()},
        }


def build_query_index(
    similarity: torch.Tensor,
    phrase_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a similarity matrix into the (scores, relevance, query) triple.

    The diagonal is excluded on both sides: a phrase retrieving itself is neither a
    hit worth crediting nor a candidate worth ranking.
    """
    diagonal = torch.eye(len(phrase_ids), dtype=torch.bool)

    scores = similarity.clone()
    scores[diagonal] = -torch.inf

    relevant = phrase_ids[:, None] == phrase_ids[None, :]
    relevant[diagonal] = False

    queries = torch.arange(len(phrase_ids)).repeat_interleave(len(phrase_ids))
    return scores.flatten(), relevant.flatten(), queries


def retrieval_scores(similarity: torch.Tensor, phrase_ids: torch.Tensor) -> RetrievalScores:
    """Evaluate a full similarity matrix as a leave-one-out retrieval task."""
    scores, relevant, queries = build_query_index(similarity, phrase_ids)
    return RetrievalScores(
        mean_average_precision=RetrievalMAP()(scores, relevant, queries).item(),
        mean_reciprocal_rank=RetrievalMRR()(scores, relevant, queries).item(),
        precision_at_k={
            k: RetrievalPrecision(top_k=k)(scores, relevant, queries).item() for k in PRECISION_CUTOFFS
        },
    )


def average_precision_per_query(similarity: torch.Tensor, phrase_ids: torch.Tensor) -> np.ndarray:
    """Average precision of each individual query, for the phrase-wise figure."""
    scores, relevant, queries = build_query_index(similarity, phrase_ids)
    metric = RetrievalMAP()
    return np.array(
        [
            metric(scores[queries == query], relevant[queries == query], queries[queries == query]).item()
            for query in range(len(phrase_ids))
        ]
    )
