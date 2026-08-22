from functools import lru_cache
import logging

from google.cloud import logging as cloud_logging


class Log:
    """Wrapper for Google Cloud Logging client."""
    def __init__(self, logger_name="frankly-match"):
        self._logger_name = logger_name

    @staticmethod
    @lru_cache(maxsize=1)
    def _client():
        return cloud_logging.Client()

    def get_trace(self, request):
        if request is None:
            return None

        # Grab the trace header from the API Gateway request
        trace_header = request.headers.get("X-Cloud-Trace-Context")
        project = self._client().project

        if trace_header and project:
            # The ID is everything before the first slash
            trace_id = trace_header.split("/")[0]
            return f"projects/{project}/traces/{trace_id}"
        return None

    def log_event(self, severity, message, request, extra_data=None):
        payload = {"message": message}
        if extra_data:
            payload.update(extra_data)

        try:
            self._client().logger(self._logger_name).log_struct(
                payload, severity=severity, trace=self.get_trace(request)
            )
        except Exception as e:
            logging.log(logging.INFO, f"Message: {message}, Extra Data: {extra_data}")
            print(f"Logging failed: {e}")

log = Log()
