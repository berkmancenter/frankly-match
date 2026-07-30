import json
import unittest

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

    def test_requires_token(self):
        client = HuggingFaceEmbeddingClient(
            endpoint_url="https://example.test",
            token="",
        )

        with self.assertRaisesRegex(EmbeddingServiceError, "HF_TOKEN"):
            client.embed(["alpha"])


if __name__ == "__main__":
    unittest.main()
