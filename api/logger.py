from functools import lru_cache
import json
import logging

from google.cloud import logging as cloud_logging


_fallback = logging.getLogger("frankly-match")
# This logger carries the diagnostics that are no longer in the API response, so
# it needs its own handler at INFO. Without one it inherits the root logger's
# WARNING level and Python's last-resort handler is WARNING-only, which would
# silently discard every INFO record exactly when Cloud Logging is unavailable.
if not _fallback.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    _fallback.addHandler(_handler)
_fallback.setLevel(logging.INFO)
_fallback.propagate = False  # our own handler emits these; do not double-log

_SEVERITY_TO_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Cloud Logging rejects/truncates a LogEntry over ~256 KiB. Leave headroom for
# the entry's own metadata (severity, trace, timestamp, insertId, ...) added
# on top of the struct payload we send.
_MAX_ENTRY_BYTES = 240_000


def _encoded_size(value):
    return len(json.dumps(value, default=str).encode("utf-8"))


def _chunk_payload(payload, max_bytes=_MAX_ENTRY_BYTES):
    """Split ``payload`` into pieces that each encode under ``max_bytes``.

    An event's extra_data can hold one unbounded list -- per-participant
    free-text responses are the current example -- that alone can push the
    encoded payload past what a single LogEntry can hold. Finds the
    list-valued field responsible for the most bytes and divides only that
    field across chunks, carrying every other field on each chunk along with
    chunk_index/chunk_count so the pieces can be correlated and reassembled
    later. Payloads with no list field, or already under the limit, are
    returned unchanged as a single-item list.
    """
    if _encoded_size(payload) <= max_bytes:
        return [payload]

    list_fields = [key for key, value in payload.items() if isinstance(value, list)]
    if not list_fields:
        return [payload]

    field = max(list_fields, key=lambda key: _encoded_size(payload[key]))
    items = payload[field]
    base = {key: value for key, value in payload.items() if key != field}

    chunks = []
    current = []
    for item in items:
        candidate = current + [item]
        # Probe with placeholder chunk_index/chunk_count so the size check
        # accounts for the metadata added to each chunk below. len(items) is a
        # safe upper bound for chunk_count -- there can never be more chunks
        # than items -- so the real (smaller) values never push a finished
        # chunk over max_bytes.
        probe = {**base, field: candidate, "chunk_index": 0, "chunk_count": len(items)}
        if current and _encoded_size(probe) > max_bytes:
            chunks.append(current)
            current = [item]
        else:
            current = candidate
    chunks.append(current)

    total = len(chunks)
    if total == 1:
        return [payload]
    return [
        {**base, field: chunk, "chunk_index": index, "chunk_count": total}
        for index, chunk in enumerate(chunks)
    ]


class Log:
    """Wrapper for Google Cloud Logging client."""
    def __init__(self, logger_name="frankly-match"):
        self._logger_name = logger_name

    @staticmethod
    @lru_cache(maxsize=1)
    def _client():
        """The Cloud Logging client, or None if it cannot be constructed.

        Returns None rather than raising so that lru_cache stores the result.
        A call that raises is not cached, so raising here would make every
        log_event redo credential discovery -- roughly three seconds each, on
        every event, for as long as credentials are unavailable.
        """
        try:
            return cloud_logging.Client()
        except Exception as exc:
            _fallback.warning(
                "Cloud Logging unavailable, falling back to stdout: %s", exc
            )
            return None

    def get_trace(self, request):
        if request is None:
            return None

        client = self._client()
        if client is None:
            return None

        # Grab the trace header from the API Gateway request
        trace_header = request.headers.get("X-Cloud-Trace-Context")
        project = client.project

        if trace_header and project:
            # The ID is everything before the first slash
            trace_id = trace_header.split("/")[0]
            return f"projects/{project}/traces/{trace_id}"
        return None

    def log_event(self, severity, message, request, extra_data=None):
        payload = {"message": message}
        if extra_data:
            payload.update(extra_data)

        level = _SEVERITY_TO_LEVEL.get(severity, logging.INFO)
        client = self._client()
        if client is None:
            _fallback.log(level, "%s | %s", message, extra_data or {})
            return

        try:
            logger = client.logger(self._logger_name)
            trace = self.get_trace(request)
            for chunk in _chunk_payload(payload):
                logger.log_struct(chunk, severity=severity, trace=trace)
        except Exception as exc:
            _fallback.log(level, "%s | %s", message, extra_data or {})
            _fallback.warning("Cloud Logging write failed: %s", exc)


log = Log()
