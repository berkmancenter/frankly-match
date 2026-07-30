from __future__ import annotations

import os
import time
from collections.abc import Sequence

import httpx
import numpy as np


DEFAULT_HF_ENDPOINT_URL = (
    "https://ttbr5oxt2cpe36qm.eu-west-1.aws.endpoints.huggingface.cloud"
)


class EmbeddingServiceError(RuntimeError):
    pass


class HuggingFaceEmbeddingClient:
    def __init__(
        self,
        endpoint_url: str | None = None,
        token: str | None = None,
        batch_size: int = 64,
        timeout_seconds: float = 60.0,
        total_timeout_seconds: float = 120.0,
        max_retries: int = 1,
        transport: httpx.BaseTransport | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if total_timeout_seconds <= 0:
            raise ValueError("total_timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        self.endpoint_url = (
            endpoint_url or os.getenv("HF_ENDPOINT_URL") or DEFAULT_HF_ENDPOINT_URL
        ).rstrip("/")
        self.token = token if token is not None else os.getenv("HF_TOKEN")
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.max_retries = max_retries
        self.transport = transport

    def embed(self, sentences: Sequence[str]) -> np.ndarray:
        texts = list(sentences)
        if not texts:
            raise EmbeddingServiceError("At least one sentence is required")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingServiceError("Every sentence must be a non-empty string")
        if not self.token:
            raise EmbeddingServiceError("HF_TOKEN is not configured")

        batches: list[np.ndarray] = []
        expected_dimensions: int | None = None
        deadline = time.monotonic() + self.total_timeout_seconds

        with httpx.Client(transport=self.transport) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                embeddings = self._request_batch(client, batch, deadline)
                if expected_dimensions is None:
                    expected_dimensions = embeddings.shape[1]
                elif embeddings.shape[1] != expected_dimensions:
                    raise EmbeddingServiceError(
                        "Embedding dimensions changed between endpoint batches"
                    )
                batches.append(embeddings)

        combined = np.vstack(batches)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingServiceError("Endpoint returned a zero-length embedding")
        return combined / norms

    def _request_batch(
        self,
        client: httpx.Client,
        sentences: list[str],
        deadline: float,
    ) -> np.ndarray:
        response: httpx.Response | None = None

        for attempt in range(self.max_retries + 1):
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise EmbeddingServiceError(
                    "Embedding request exceeded its total timeout"
                )
            try:
                response = client.post(
                    self.endpoint_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                    },
                    json={"inputs": sentences},
                    timeout=httpx.Timeout(
                        min(self.timeout_seconds, remaining_seconds)
                    ),
                )
            except httpx.RequestError as exc:
                if attempt == self.max_retries:
                    raise EmbeddingServiceError(
                        "Could not reach the embedding endpoint"
                    ) from exc
                self._wait_before_retry(attempt, deadline)
                continue

            if response.status_code < 400:
                break
            if (
                response.status_code != 429 and response.status_code < 500
            ) or attempt == self.max_retries:
                raise EmbeddingServiceError(
                    f"Embedding endpoint returned HTTP {response.status_code}"
                )
            self._wait_before_retry(attempt, deadline)

        if response is None:
            raise EmbeddingServiceError("Embedding endpoint returned no response")

        try:
            payload = response.json()
            raw_embeddings = payload["embeddings"]
            embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingServiceError(
                "Embedding endpoint returned an invalid response"
            ) from exc

        if embeddings.ndim != 2:
            raise EmbeddingServiceError(
                "Embedding endpoint response must be a two-dimensional matrix"
            )
        if embeddings.shape[0] != len(sentences):
            raise EmbeddingServiceError(
                "Embedding endpoint returned the wrong number of vectors"
            )
        if embeddings.shape[1] == 0:
            raise EmbeddingServiceError("Embedding vectors must not be empty")
        if not np.isfinite(embeddings).all():
            raise EmbeddingServiceError(
                "Embedding endpoint returned non-finite values"
            )
        return embeddings

    @staticmethod
    def _wait_before_retry(attempt: int, deadline: float) -> None:
        delay = min(0.25 * (2**attempt), max(0.0, deadline - time.monotonic()))
        if delay > 0:
            time.sleep(delay)
