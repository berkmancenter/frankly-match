from functools import lru_cache
from contextvars import ContextVar
import json
import logging

from google.cloud import logging as cloud_logging


log_context: ContextVar[dict] = ContextVar("match_log_context", default={})

LOG_SCHEMA_VERSION = 1


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


def _with_context(payload: dict) -> dict:
    """Attach the request context and schema version to a payload."""
    return {**payload, **log_context.get(), "log_schema_version": LOG_SCHEMA_VERSION}


def _emit_fallback(level: int, payload: dict) -> None:
    """Write one structured record to stderr.

    Every stderr record goes through here, diagnostics included, so it carries
    the same match_run_id and schema version as the Cloud Logging payloads and
    stays joinable to the request that produced it.
    """
    _fallback.log(level, "%s", json.dumps(_with_context(payload), ensure_ascii=False))


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
            # The None is cached, so this fires once per process and every
            # later log_event in the process lands on stderr as well -- not
            # only the request whose run ID happens to be attached here.
            _emit_fallback(logging.WARNING, {
                "message": "Cloud Logging unavailable; this process is falling back to stderr",
                "scope": "process",
                "error": str(exc),
            })
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
            _emit_fallback(level, payload)
            return

        try:
            client.logger(self._logger_name).log_struct(
                _with_context(payload), severity=severity, trace=self.get_trace(request)
            )
        except Exception as exc:
            _emit_fallback(level, payload)
            _emit_fallback(logging.WARNING, {
                "message": "Cloud Logging write failed; payload preserved on stderr",
                "failed_message": message,
                "error": str(exc),
            })


log = Log()
