"""Pre-survey data: the committed comment catalog and the private approval matrix.

The catalog (data/presurvey_comments.json) ships with the code: every comment
written in the pre-survey, its topic, its author's pseudonymous pid and its
embedding. It holds no names or emails, so it can live in the public repo, and
because it ships with the image it cannot fail to load at runtime.

The approval matrix holds names and emails, so it lives in GCS and is fetched
per request, after groups are final. Rows are pre-survey voters, columns are
catalog comments, and each cell is the probability that voter approves that
comment: the observed vote (exactly 0 or 1) where one was cast, a model
prediction otherwise.

The matrix is also the only bridge between the two surveys: a registrant's
email or name finds their row, and the row's pid finds the comments they wrote.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

# The registration question whose answers pick the diffusion comment, and the
# topic the comment is drawn from.
DIFFUSION_TOPIC_ID = "stocking_growing"
CATALOG_PATH = Path(__file__).resolve().parent / "data" / "presurvey_comments.json"
# Offline LLM screen of the diffusion topic's comments; see
# scripts/judge_diffusion_comments.py. Only comments marked eligible are shown.
ELIGIBILITY_PATH = Path(__file__).resolve().parent / "data" / "diffusion_eligibility.json"
# Eligible comments ranked by bridging over all pre-survey voters; see
# scripts/rank_global_bridging.py. The fallback for any statement a table
# cannot compute.
BRIDGING_RANKING_PATH = Path(__file__).resolve().parent / "data" / "bridging_ranking.json"
CATALOG_SCHEMA_VERSION = 1
MATRIX_IDENTITY_COLUMNS = ("email", "name", "pid")
UNIT_NORM_TOLERANCE = 1e-3
# The matrix is read after groups are final, but the request is still open:
# a slow read delays the response Frankly is waiting on. The library's own
# retry would keep retrying transient errors for up to two minutes, so it is
# off, and a short bounded retry of our own replaces it: at worst about
# GCS_ATTEMPTS * GCS_TIMEOUT_SECONDS before bridging falls back.
GCS_TIMEOUT_SECONDS = 10.0
GCS_ATTEMPTS = 2

LinkMethod = Literal["email", "name", "none"]


class PreSurveyDataError(ValueError):
    """A pre-survey file is malformed or inconsistent with the catalog."""


@dataclass(frozen=True)
class PreSurveyComment:
    comment_id: str
    text: str
    topic_id: str
    author_pid: str


@dataclass(frozen=True)
class CommentCatalog:
    comments: tuple[PreSurveyComment, ...]
    # Row i is the unit-length embedding of comments[i].
    embeddings: np.ndarray
    topic_ids: tuple[str, ...]
    embedding_model: str | None
    embedding_revision: str | None
    sha256: str

    def topic_counts(self) -> dict[str, int]:
        counts = {topic: 0 for topic in self.topic_ids}
        for comment in self.comments:
            counts[comment.topic_id] += 1
        return counts


@dataclass(frozen=True)
class ApprovalMatrix:
    pids: tuple[str, ...]
    emails: tuple[str, ...]
    names: tuple[str, ...]
    # voters x comments, columns in catalog order.
    probabilities: np.ndarray
    source: str
    sha256: str

    @property
    def observed_share(self) -> float:
        """Share of cells that are exactly 0 or 1, read as observed votes.

        The file does not mark which cells were observed. A prediction can also
        land on exactly 0 or 1, so this is a summary for the logs, never an
        input to any decision.
        """
        values = self.probabilities
        return float(np.mean((values == 0.0) | (values == 1.0)))


@dataclass(frozen=True)
class Link:
    participant_id: str
    method: LinkMethod
    row: int | None
    presurvey_pid: str | None


def parse_comment_catalog(raw: bytes) -> CommentCatalog:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreSurveyDataError(f"catalog is not valid JSON: {exc}") from exc
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise PreSurveyDataError(
            f"catalog schema_version must be {CATALOG_SCHEMA_VERSION}"
        )

    topic_ids = tuple(document.get("topics") or {})
    if DIFFUSION_TOPIC_ID not in topic_ids:
        raise PreSurveyDataError(
            f"catalog has no '{DIFFUSION_TOPIC_ID}' topic"
        )
    metadata = document.get("embedding_metadata") or {}
    dimensions = metadata.get("dimensions")

    comments: list[PreSurveyComment] = []
    vectors: list[list[float]] = []
    for key, entry in (document.get("comments") or {}).items():
        if entry.get("comment_id") != key:
            raise PreSurveyDataError(f"comment {key} has a mismatched comment_id")
        if entry.get("topic_id") not in topic_ids:
            raise PreSurveyDataError(f"comment {key} has an unknown topic_id")
        text = entry.get("text")
        author = entry.get("author_pid")
        if not isinstance(text, str) or not text.strip():
            raise PreSurveyDataError(f"comment {key} has no text")
        if not isinstance(author, str) or not author:
            raise PreSurveyDataError(f"comment {key} has no author_pid")
        embedding = entry.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != dimensions:
            raise PreSurveyDataError(
                f"comment {key} embedding does not have {dimensions} dimensions"
            )
        comments.append(PreSurveyComment(key, text, entry["topic_id"], author))
        vectors.append(embedding)

    if not comments:
        raise PreSurveyDataError("catalog has no comments")
    embeddings = np.asarray(vectors, dtype=np.float64)
    if not np.isfinite(embeddings).all():
        raise PreSurveyDataError("catalog embeddings must be finite")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE):
        raise PreSurveyDataError("catalog embeddings must be unit length")

    return CommentCatalog(
        comments=tuple(comments),
        embeddings=embeddings,
        topic_ids=topic_ids,
        embedding_model=metadata.get("model"),
        embedding_revision=metadata.get("revision"),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


@lru_cache(maxsize=1)
def load_comment_catalog() -> CommentCatalog:
    return parse_comment_catalog(CATALOG_PATH.read_bytes())


@lru_cache(maxsize=1)
def load_eligible_comment_ids() -> frozenset[str]:
    document = json.loads(ELIGIBILITY_PATH.read_bytes())
    return frozenset(
        comment_id
        for comment_id, entry in document["comments"].items()
        if entry["eligible"]
    )


@lru_cache(maxsize=1)
def load_bridging_ranking() -> tuple[str, ...]:
    document = json.loads(BRIDGING_RANKING_PATH.read_bytes())
    return tuple(entry["comment_id"] for entry in document["ranking"])


def parse_approval_matrix(
    raw: bytes, catalog: CommentCatalog, source: str
) -> ApprovalMatrix:
    """Parse the CSV and reorder its comment columns into catalog order.

    The comment columns must be exactly the catalog's comment ids. A matrix
    built against a different catalog would silently misattribute every vote,
    so any mismatch rejects the whole file.
    """
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PreSurveyDataError(f"approval matrix is not UTF-8: {exc}") from exc
    reader = csv.reader(io.StringIO(text))
    try:
        header = [column.strip() for column in next(reader)]
    except StopIteration as exc:
        raise PreSurveyDataError("approval matrix is empty") from exc

    missing = [c for c in MATRIX_IDENTITY_COLUMNS if c not in header]
    if missing:
        raise PreSurveyDataError(f"approval matrix lacks columns: {missing}")
    identity_index = {c: header.index(c) for c in MATRIX_IDENTITY_COLUMNS}
    comment_columns = {
        column: index for index, column in enumerate(header)
        if column not in MATRIX_IDENTITY_COLUMNS
    }
    catalog_ids = [comment.comment_id for comment in catalog.comments]
    if set(comment_columns) != set(catalog_ids) or len(comment_columns) != len(header) - 3:
        extra = sorted(set(comment_columns) - set(catalog_ids))[:5]
        absent = sorted(set(catalog_ids) - set(comment_columns))[:5]
        raise PreSurveyDataError(
            "approval matrix columns do not match the comment catalog "
            f"(unknown: {extra}, missing: {absent})"
        )
    order = [comment_columns[comment_id] for comment_id in catalog_ids]

    pids: list[str] = []
    emails: list[str] = []
    names: list[str] = []
    rows: list[list[float]] = []
    for line_number, record in enumerate(reader, start=2):
        if not any(cell.strip() for cell in record):
            continue
        if len(record) != len(header):
            raise PreSurveyDataError(
                f"approval matrix line {line_number} has {len(record)} cells, "
                f"expected {len(header)}"
            )
        try:
            rows.append([float(record[index]) for index in order])
        except ValueError as exc:
            raise PreSurveyDataError(
                f"approval matrix line {line_number} has a non-numeric cell"
            ) from exc
        pids.append(record[identity_index["pid"]].strip())
        emails.append(record[identity_index["email"]].strip())
        names.append(record[identity_index["name"]].strip())

    if not rows:
        raise PreSurveyDataError("approval matrix has no voter rows")
    if len(set(pids)) != len(pids) or not all(pids):
        raise PreSurveyDataError("approval matrix pids must be present and unique")
    probabilities = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise PreSurveyDataError("approval matrix values must lie in [0, 1]")

    return ApprovalMatrix(
        pids=tuple(pids),
        emails=tuple(emails),
        names=tuple(names),
        probabilities=probabilities,
        source=source,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_source(uri: str) -> bytes:
    """Read gs://bucket/object via the ambient service account, else a local path."""
    if uri.startswith("gs://"):
        bucket, _, blob = uri[len("gs://"):].partition("/")
        if not bucket or not blob:
            raise PreSurveyDataError(f"'{uri}' is not a gs://bucket/object path")
        return _download_gcs(bucket, blob)
    return Path(uri).expanduser().read_bytes()


def _download_gcs(bucket: str, blob: str) -> bytes:
    # Imported here so tests and local runs never need the package or credentials.
    from google.cloud import storage

    target = storage.Client().bucket(bucket).blob(blob)
    for attempt in range(GCS_ATTEMPTS):
        try:
            return target.download_as_bytes(timeout=GCS_TIMEOUT_SECONDS, retry=None)
        except Exception:
            if attempt == GCS_ATTEMPTS - 1:
                raise


def load_approval_matrix(uri: str, catalog: CommentCatalog) -> ApprovalMatrix:
    return parse_approval_matrix(read_source(uri), catalog, uri)


def normalize_email(value: str) -> str:
    return value.strip().casefold()


def normalize_name(value: str) -> str:
    """Case, accents and spacing are ignored: 'José  García' == 'jose garcia'."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


def _unique_index(values: tuple[str, ...], normalize) -> dict[str, int]:
    """Normalised value -> row, keeping only values that name exactly one row."""
    rows: dict[str, list[int]] = {}
    for row, value in enumerate(values):
        key = normalize(value)
        if key:
            rows.setdefault(key, []).append(row)
    return {key: found[0] for key, found in rows.items() if len(found) == 1}


def link_participants(
    identities: dict[str, tuple[str | None, str | None]],
    matrix: ApprovalMatrix,
) -> dict[str, Link]:
    """Find each registrant's pre-survey row: by email, then by name.

    Either key is used only when it names exactly one row, so an ambiguous
    match is never guessed. Someone found by neither is unmatched, which is
    expected for anyone who registered without taking the pre-survey.
    """
    by_email = _unique_index(matrix.emails, normalize_email)
    by_name = _unique_index(matrix.names, normalize_name)
    links: dict[str, Link] = {}
    for participant_id, (email, name) in identities.items():
        method: LinkMethod = "none"
        row = None
        if email and normalize_email(email) in by_email:
            method, row = "email", by_email[normalize_email(email)]
        elif name and normalize_name(name) in by_name:
            method, row = "name", by_name[normalize_name(name)]
        links[participant_id] = Link(
            participant_id=participant_id,
            method=method,
            row=row,
            presurvey_pid=matrix.pids[row] if row is not None else None,
        )
    return links
