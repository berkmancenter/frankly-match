import json
import unittest
from unittest.mock import Mock, patch

from logger import Log, log_context


class LoggerTests(unittest.TestCase):
    def test_cloud_and_fallback_preserve_context_and_payload(self):
        client = Mock()
        token = log_context.set({"match_run_id": "run-1", "study_id": "study-1"})
        try:
            for fail in (False, True):
                client.logger.return_value.log_struct.side_effect = RuntimeError("offline") if fail else None
                with patch.object(Log, "_client", return_value=client), patch("logger._fallback") as fallback:
                    Log().log_event("INFO", "Example", None, {"distances": [0.0, 0.5]})
                    payload = client.logger.return_value.log_struct.call_args.args[0]
                    self.assertEqual(payload["match_run_id"], "run-1")
                    self.assertEqual(payload["study_id"], "study-1")
                    self.assertEqual(payload["log_schema_version"], 1)
                    if fail:
                        self.assertEqual(json.loads(fallback.log.call_args.args[2]), payload)
        finally:
            log_context.reset(token)

    def test_no_cloud_client_keeps_run_id(self):
        token = log_context.set({"match_run_id": "local-run"})
        try:
            with patch.object(Log, "_client", return_value=None), patch("logger._fallback") as fallback:
                Log().log_event("WARNING", "Example", None)
                self.assertEqual(json.loads(fallback.log.call_args.args[2])["match_run_id"], "local-run")
        finally:
            log_context.reset(token)
