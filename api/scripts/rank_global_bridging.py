"""Rank the diffusion candidates by bridging across the whole pre-survey.

Scores every eligible diffusion-topic comment with the same probabilistic
pairwise-disagreement score used per table, but with every pre-survey voter as
the population. The ranking is the fallback whenever a table's own statement
cannot be computed, including when the approval matrix is unavailable at
runtime, which is why it is computed here and committed.

Usage, from api/:
    .venv/bin/python scripts/rank_global_bridging.py ../approval_matrix.csv

Writes data/bridging_ranking.json: comment ids and scores only, no personal data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from bridging import pairwise_disagreement_scores  # noqa: E402
from presurvey import (  # noqa: E402
    BRIDGING_RANKING_PATH,
    DIFFUSION_TOPIC_ID,
    load_approval_matrix,
    load_comment_catalog,
    load_eligible_comment_ids,
)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    catalog = load_comment_catalog()
    matrix = load_approval_matrix(sys.argv[1], catalog)
    eligible = load_eligible_comment_ids()
    rows = [
        row for row, comment in enumerate(catalog.comments)
        if comment.topic_id == DIFFUSION_TOPIC_ID and comment.comment_id in eligible
    ]
    scores = pairwise_disagreement_scores(matrix.probabilities, rows)
    ranked = sorted(zip(rows, scores), key=lambda pair: -pair[1])
    document = {
        "description": (
            "Eligible diffusion-topic comments ranked by probabilistic pairwise "
            "disagreement with every pre-survey voter as the population. Used as "
            "the fallback for any statement a table cannot compute."
        ),
        "voter_count": len(matrix.pids),
        "matrix_sha256": matrix.sha256,
        "catalog_sha256": catalog.sha256,
        "ranking": [
            {"comment_id": catalog.comments[row].comment_id, "score": float(score)}
            for row, score in ranked
        ],
    }
    BRIDGING_RANKING_PATH.write_text(json.dumps(document, indent=2) + "\n")
    print(f"ranked {len(ranked)} comments over {len(matrix.pids)} voters")
    for row, score in ranked[:5]:
        print(f"  {score:.4f}  {' '.join(catalog.comments[row].text.split())[:100]}")


if __name__ == "__main__":
    main()
