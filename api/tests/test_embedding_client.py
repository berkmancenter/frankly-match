import json
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from embedding_client import EmbeddingServiceError, HuggingFaceEmbeddingClient


class HuggingFaceEmbeddingClientTests(unittest.TestCase):
    def test_batches_and_normalizes_vectors(self):
        request_sizes = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            request_sizes.append(len(payload["inputs"]))
            embeddings = [
                [float(len(text)), float(index + 1)]
                for index, text in enumerate(payload["inputs"])
            ]
            return httpx.Response(200, json={"embeddings": embeddings})

        client = HuggingFaceEmbeddingClient(
            endpoint_url="https://example.test",
            token="test-token",
            batch_size=2,
            transport=httpx.MockTransport(handler),
        )

        embeddings = client.embed(["alpha", "beta", "gamma"])

        self.assertEqual(request_sizes, [2, 1])
        self.assertEqual(embeddings.shape, (3, 2))
        np.testing.assert_allclose(
            np.linalg.norm(embeddings, axis=1),
            np.ones(3),
        )

    def test_rejects_malformed_endpoint_response(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"unexpected": []})
        )
        client = HuggingFaceEmbeddingClient(
            endpoint_url="https://example.test",
            token="test-token",
            transport=transport,
        )

        with self.assertRaises(EmbeddingServiceError):
            client.embed(["alpha"])

    def test_waits_out_cold_start_503_then_succeeds(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] <= 2:
                return httpx.Response(
                    503, json={"error": "loading", "estimated_time": 1.5}
                )
            payload = json.loads(request.content)
            embeddings = [[1.0, 0.0] for _ in payload["inputs"]]
            return httpx.Response(200, json={"embeddings": embeddings})

        client = HuggingFaceEmbeddingClient(
            endpoint_url="https://example.test",
            token="test-token",
            max_retries=0,
            transport=httpx.MockTransport(handler),
        )

        with patch("embedding_client.time.sleep") as mock_sleep:
            embeddings = client.embed(["alpha"])

        self.assertEqual(calls["count"], 3)
        self.assertEqual(embeddings.shape, (1, 2))
        self.assertEqual(mock_sleep.call_count, 2)
        for call in mock_sleep.call_args_list:
            self.assertEqual(call.args[0], 1.5)

    def test_cold_start_503_past_deadline_raises(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "loading"})
        )
        client = HuggingFaceEmbeddingClient(
            endpoint_url="https://example.test",
            token="test-token",
            total_timeout_seconds=0.2,
            transport=transport,
        )

        with patch("embedding_client.time.sleep"):
            with self.assertRaisesRegex(EmbeddingServiceError, "total timeout"):
                client.embed(["alpha"])

    def test_requires_token(self):
        client = HuggingFaceEmbeddingClient(
            endpoint_url="https://example.test",
            token="",
        )

        with self.assertRaisesRegex(EmbeddingServiceError, "HF_TOKEN"):
            client.embed(["alpha"])


if __name__ == "__main__":
    unittest.main()
