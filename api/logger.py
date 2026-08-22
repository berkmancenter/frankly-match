from functools import lru_cache
import logging

from google.cloud import logging as cloud_logging


_fallback = logging.getLogger("frankly-match")

_SEVERITY_TO_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


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
            client.logger(self._logger_name).log_struct(
                payload, severity=severity, trace=self.get_trace(request)
            )
        except Exception as exc:
            _fallback.log(level, "%s | %s", message, extra_data or {})
            _fallback.warning("Cloud Logging write failed: %s", exc)


log = Log()
