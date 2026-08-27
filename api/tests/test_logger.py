"""Large event payloads must be chunked before being sent to Cloud Logging.

log_event used to send extra_data as one log_struct call regardless of size.
Cloud Logging rejects/truncates a LogEntry over ~256 KiB, so an event whose
extra_data holds an unbounded list -- per-participant free-text responses,
for example -- could lose its only record instead of being split.
"""
import unittest
from unittest.mock import MagicMock, patch

from logger import Log, _chunk_payload, _encoded_size


class ChunkPayloadTests(unittest.TestCase):
    def test_small_payload_returned_as_single_chunk(self):
        payload = {"message": "hi", "responses": [1, 2, 3]}
        self.assertEqual(_chunk_payload(payload, max_bytes=10_000), [payload])

    def test_large_list_field_is_split_across_chunks(self):
        payload = {
            "message": "collected responses",
            "participant_count": 20,
            "responses": [{"id": str(i), "text": "x" * 50} for i in range(20)],
        }
        # Small enough that the full list can't fit in one chunk, large enough
        # that more than one item fits per chunk.
        max_bytes = 300

        chunks = _chunk_payload(payload, max_bytes=max_bytes)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(_encoded_size(chunk), max_bytes)
            self.assertEqual(chunk["message"], "collected responses")
            self.assertEqual(chunk["participant_count"], 20)
            self.assertEqual(chunk["chunk_count"], len(chunks))

        # Every response survives, in order, across the reassembled chunks.
        reassembled = [item for chunk in chunks for item in chunk["responses"]]
        self.assertEqual(reassembled, payload["responses"])
        self.assertEqual(
            [chunk["chunk_index"] for chunk in chunks], list(range(len(chunks)))
        )

    def test_no_list_field_returns_payload_unchanged(self):
        payload = {"message": "big string", "blob": "x" * 1000}
        chunks = _chunk_payload(payload, max_bytes=100)
        self.assertEqual(chunks, [payload])

    def test_splits_the_largest_list_field_only(self):
        payload = {
            "message": "two lists",
            "small_list": ["a", "b"],
            "responses": [{"text": "y" * 50} for _ in range(20)],
        }
        chunks = _chunk_payload(payload, max_bytes=300)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["small_list"], ["a", "b"])
        reassembled = [item for chunk in chunks for item in chunk["responses"]]
        self.assertEqual(reassembled, payload["responses"])

    def test_single_oversized_item_is_kept_whole(self):
        payload = {"message": "one huge item", "responses": ["x" * 1000]}
        chunks = _chunk_payload(payload, max_bytes=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["responses"], ["x" * 1000])

    def test_falls_back_to_unchunked_when_one_item_cannot_share_a_chunk(self):
        # A single item too large to fit alongside anything else forces its own
        # chunk, which can still be over max_bytes -- and unlike the two normal
        # chunks around it, there is no smaller grouping that would fix it.
        payload = {
            "message": "mixed sizes",
            "responses": ["a" * 50, "b" * 50, "c" * 1000, "d" * 50],
        }
        max_bytes = 300

        chunks = _chunk_payload(payload, max_bytes=max_bytes)

        self.assertEqual(chunks, [payload])

    def test_falls_back_to_unchunked_when_base_fields_alone_exceed_the_budget(self):
        payload = {
            "message": "m",
            "context": "z" * 1000,  # non-list, carried on every chunk
            "responses": [{"text": "y" * 50} for _ in range(20)],
        }
        max_bytes = 300

        chunks = _chunk_payload(payload, max_bytes=max_bytes)

        self.assertEqual(chunks, [payload])


class LogEventChunkingTests(unittest.TestCase):
    def test_log_event_writes_one_log_struct_call_per_chunk(self):
        log = Log()
        fake_client = MagicMock()
        fake_logger = MagicMock()
        fake_client.logger.return_value = fake_logger

        payload_chunks = [
            {"message": "m", "chunk_index": 0, "chunk_count": 2},
            {"message": "m", "chunk_index": 1, "chunk_count": 2},
        ]

        with patch.object(Log, "_client", staticmethod(lambda: fake_client)), patch(
            "logger._chunk_payload", return_value=payload_chunks
        ):
            log.log_event("INFO", "m", request=None, extra_data={"responses": []})

        self.assertEqual(fake_logger.log_struct.call_count, 2)
        seen_chunks = [call.args[0] for call in fake_logger.log_struct.call_args_list]
        self.assertEqual(seen_chunks, payload_chunks)


if __name__ == "__main__":
    unittest.main()
