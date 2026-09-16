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
                        records = [json.loads(call.args[2]) for call in fallback.log.call_args_list]
                        self.assertEqual(records[0], payload)
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

    def test_write_failure_diagnostic_is_structured_and_joinable(self):
        """The diagnostic emitted next to the preserved payload used to be a
        plain-text warning with no run ID, so it could not be correlated with
        the request whose write had failed."""
        client = Mock()
        client.logger.return_value.log_struct.side_effect = RuntimeError("offline")
        token = log_context.set({"match_run_id": "run-2"})
        try:
            with patch.object(Log, "_client", return_value=client), patch("logger._fallback") as fallback:
                Log().log_event("INFO", "Example", None, {"k": 1})
                records = [json.loads(call.args[2]) for call in fallback.log.call_args_list]
        finally:
            log_context.reset(token)

        self.assertEqual(len(records), 2)
        payload, diagnostic = records
        self.assertEqual(payload["k"], 1)
        self.assertIn("write failed", diagnostic["message"])
        self.assertEqual(diagnostic["failed_message"], "Example")
        self.assertEqual(diagnostic["error"], "offline")
        for record in records:
            self.assertEqual(record["match_run_id"], "run-2")
            self.assertEqual(record["log_schema_version"], 1)
        fallback.warning.assert_not_called()

    def test_unavailable_client_diagnostic_is_structured_and_process_wide(self):
        """_client caches its None, so the credential warning fires once per
        process. It must say so rather than imply only one run was affected."""
        Log._client.cache_clear()
        token = log_context.set({"match_run_id": "run-3"})
        try:
            with patch("logger.cloud_logging.Client", side_effect=RuntimeError("no credentials")), patch(
                "logger._fallback"
            ) as fallback:
                self.assertIsNone(Log._client())
                record = json.loads(fallback.log.call_args.args[2])
        finally:
            log_context.reset(token)
            Log._client.cache_clear()

        self.assertEqual(record["scope"], "process")
        self.assertEqual(record["error"], "no credentials")
        self.assertEqual(record["match_run_id"], "run-3")
        self.assertEqual(record["log_schema_version"], 1)
        fallback.warning.assert_not_called()
