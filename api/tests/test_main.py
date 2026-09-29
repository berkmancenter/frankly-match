import unittest
from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from text_match import TextMatchGroup


class FakeTextMatchingService:
    def __init__(self):
        self.participant_responses = None
        self.target_group_size = None

    def match(self, participant_responses, target_group_size, request=None, identities=None):
        self.request = request
        self.identities = identities
        self.participant_responses = participant_responses
        self.target_group_size = target_group_size
        return [
            TextMatchGroup(
                participant_ids=list(participant_responses),
                diversity_level="medium",
                diffusion_statement="A test statement",
                fallback_used=False,
                assigned_target=0.5,
                achieved_diversity=0.62,
            )
        ]


class MatchApiTests(unittest.TestCase):
    def test_run_ids_join_logs_to_response_and_isolate_requests(self):
        from logger import Log
        import json
        service = FakeTextMatchingService()
        with patch.object(main, "get_text_matching_service", return_value=service), patch.object(
            Log, "_client", return_value=None
        ), patch("logger._fallback") as fallback:
            for event_id in ("event-a", "event-b"):
                fallback.reset_mock()
                response = self.client.post("/match", json={
                    "algorithm": "textGroupMatch", "targetGroupSize": 3,
                    "studyId": "study-1", "eventId": event_id,
                    "participants": {"a": {}, "b": {}, "c": {}},
                })
                self.assertEqual(response.status_code, 200)
                run_id = response.headers["X-Match-Run-ID"]
                records = [json.loads(call.args[2]) for call in fallback.log.call_args_list]
                self.assertTrue(all(r["match_run_id"] == run_id for r in records))
                matched = next(r for r in records if "groups" in r)
                self.assertEqual(matched["event_id"], event_id)
                self.assertEqual(matched["study_id"], "study-1")
                if event_id == "event-a":
                    first_run = run_id
                else:
                    self.assertNotEqual(first_run, run_id)

    def test_invalid_request_has_run_id(self):
        response = self.client.post("/match", json={})
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.headers["X-Match-Run-ID"])

    def setUp(self):
        self.client = TestClient(main.app)

    def test_unhandled_error_returns_run_id_and_structured_500(self):
        """Starlette's outer error middleware builds the 500 itself, without
        our header. A failed attempt must still be identifiable, and the
        traceback must not be lost once the exception no longer propagates."""
        from unittest.mock import Mock
        service = Mock()
        service.match.side_effect = RuntimeError("optimizer exploded")
        with patch.object(main, "get_text_matching_service", return_value=service), patch.object(
            main.log, "log_event"
        ) as log_event:
            response = self.client.post("/match", json={
                "algorithm": "textGroupMatch", "targetGroupSize": 3,
                "participants": {"a": {}, "b": {}, "c": {}},
            })
        self.assertEqual(response.status_code, 500)
        self.assertTrue(response.headers["X-Match-Run-ID"])
        self.assertEqual(
            response.json(), {"code": "INTERNAL_ERROR", "message": "Unexpected server error"}
        )
        failed = next(
            call for call in log_event.call_args_list if call.args[1] == "Match request failed"
        )
        self.assertEqual(failed.args[0], "ERROR")
        self.assertIn("optimizer exploded", failed.kwargs["extra_data"]["traceback"])
        self.assertEqual(failed.kwargs["extra_data"]["status_code"], 500)

    def test_strict_mode_refusal_keeps_study_linkage(self):
        """The sync route sets the ContextVar in a worker thread's copy of the
        context, which the async exception handler and the middleware's
        completion log never see. Linkage rides on request.state instead."""
        from logger import Log
        import json
        with patch.object(Log, "_client", return_value=None), patch("logger._fallback") as fallback, patch.dict(
            main.os.environ, {"REQUIRE_REAL_TEXT": "1"}
        ):
            response = self.client.post("/match", json={
                "algorithm": "textGroupMatch", "targetGroupSize": 3,
                "studyId": "study-1", "eventId": "event-x",
                "participants": {"a": {"freeTextResponse": "real"}, "b": {}, "c": {}},
            })
        self.assertEqual(response.status_code, 422)
        run_id = response.headers["X-Match-Run-ID"]
        records = [json.loads(call.args[2]) for call in fallback.log.call_args_list]
        refusal = next(r for r in records if r["message"].startswith("Refusing to match"))
        completed = next(r for r in records if r["message"] == "Match request completed")
        for record in (refusal, completed):
            self.assertEqual(record["match_run_id"], run_id)
            self.assertEqual(record["study_id"], "study-1")
            self.assertEqual(record["event_id"], "event-x")
        self.assertEqual(completed["status_code"], 422)

    def test_validation_errors_log_with_the_request(self):
        with patch.object(main.log, "log_event") as log_event:
            response = self.client.post("/match", json={
                "algorithm": "nope", "targetGroupSize": 3, "participants": {"a": {}},
            })
        self.assertEqual(response.status_code, 400)
        error = next(
            call for call in log_event.call_args_list
            if call.args[1].startswith("Error UNKNOWN_ALGORITHM")
        )
        self.assertIsNotNone(error.kwargs["request"])

    def test_openapi_declares_run_id_header_on_every_match_response(self):
        """The spec doubles as the API Gateway config: a header missing from
        the error responses is invisible to generated clients exactly where
        they need it to identify a failed attempt."""
        import yaml
        from pathlib import Path
        spec = yaml.safe_load((Path(main.__file__).parent / "openapi.yaml").read_text())
        responses = spec["paths"]["/match"]["post"]["responses"]
        self.assertEqual(set(responses), {"200", "400", "422", "500"})
        for status, response in responses.items():
            if "$ref" in response:
                response = spec["responses"][response["$ref"].rsplit("/", 1)[-1]]
            self.assertIn("X-Match-Run-ID", response.get("headers", {}), f"{status} response")

    def test_binary_response_shape_remains_unchanged(self):
        response = self.client.post(
            "/match",
            json={
                "algorithm": "binaryGroupMatch",
                "targetGroupSize": 2,
                "participants": {
                    "a": {"binaryAnswerMask": "000"},
                    "b": {"binaryAnswerMask": "111"},
                    "c": {"binaryAnswerMask": "001"},
                    "d": {"binaryAnswerMask": "110"},
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        for group in response.json()["results"]:
            self.assertEqual(set(group), {"groupId", "participantIds"})
        self.assertEqual(set(response.json()), {"results"})

    def test_text_algorithm_returns_extended_group_fields(self):
        service = FakeTextMatchingService()
        with patch.object(
            main,
            "get_text_matching_service",
            return_value=service,
        ):
            response = self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {
                        "a": {"freeTextResponse": "  supplied response  "},
                        "b": {},
                        "c": {},
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.target_group_size, 3)
        self.assertEqual(service.participant_responses["a"], "supplied response")
        self.assertTrue(service.participant_responses["b"])
        self.assertTrue(service.participant_responses["c"])
        self.assertEqual(
            response.json()["results"][0],
            {
                "groupId": "1",
                "participantIds": ["a", "b", "c"],
                "diversityLevel": "medium",
                "diffusionStatement": "A test statement",
            },
        )

    def test_embedded_text_is_logged_rather_than_returned(self):
        """Participants who send no text get a placeholder. The caller used to
        need that echoed back; it now goes to the logger instead, which is the
        single biggest payload saving on a large event."""
        service = FakeTextMatchingService()
        with patch.object(
            main, "get_text_matching_service", return_value=service
        ), patch.object(main.log, "log_event") as log_event:
            response = self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {
                        "a": {"freeTextResponse": "  supplied response  "},
                        "b": {},
                        "c": {},
                    },
                },
            )

        self.assertNotIn("participantResponses", response.json())

        logged = next(
            call.kwargs["extra_data"]
            for call in log_event.call_args_list
            if "responses" in call.kwargs.get("extra_data", {})
        )
        by_id = {entry["participant_id"]: entry for entry in logged["responses"]}
        self.assertEqual(by_id["a"]["response"], "supplied response")
        self.assertFalse(by_id["a"]["is_placeholder"])
        self.assertTrue(by_id["b"]["is_placeholder"])
        self.assertEqual(logged["placeholder_count"], 2)

    def test_identity_is_trimmed_logged_and_never_rejects(self):
        """email/name link people to the pre-survey. A missing, blank or
        malformed value must still produce groups, never a 422."""
        service = FakeTextMatchingService()
        with patch.object(
            main, "get_text_matching_service", return_value=service
        ), patch.object(main.log, "log_event") as log_event:
            response = self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {
                        "a": {"email": "  Alice@Example.org ", "name": " Alice "},
                        "b": {"email": "not-an-email", "name": "   "},
                        "c": {"email": 123, "name": {"first": "x"}},
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        logged = next(
            call.kwargs["extra_data"]
            for call in log_event.call_args_list
            if "responses" in call.kwargs.get("extra_data", {})
        )
        by_id = {entry["participant_id"]: entry for entry in logged["responses"]}
        # Trimmed only; case is left alone here and normalised when linking.
        self.assertEqual(by_id["a"]["email"], "Alice@Example.org")
        self.assertEqual(by_id["a"]["name"], "Alice")
        self.assertEqual(by_id["b"]["email"], "not-an-email")
        self.assertIsNone(by_id["b"]["name"])
        self.assertIsNone(by_id["c"]["email"])
        self.assertIsNone(by_id["c"]["name"])

    def test_per_statement_fallback_reasons_are_logged_not_returned(self):
        """A routine per-table fallback is logged as such, not as a
        group-level failure, and nothing extra reaches the client."""
        service = FakeTextMatchingService()
        original = service.match

        def match(*args, **kwargs):
            (group,) = original(*args, **kwargs)
            return [replace(group, fallback_used=True,
                            bridging_fallback_reason="fewer_than_two_linked")]

        service.match = match
        with patch.object(
            main, "get_text_matching_service", return_value=service
        ), patch.object(main.log, "log_event") as log_event:
            response = self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {"a": {}, "b": {}, "c": {}},
                },
            )

        body = response.json()["results"][0]
        for field in ("fallbackReason", "maximinFallbackReason", "bridgingFallbackReason"):
            self.assertNotIn(field, body)
        logged_group = next(
            call.kwargs["extra_data"]["groups"][0]
            for call in log_event.call_args_list
            if call.kwargs.get("extra_data", {}).get("algorithm") == "textGroupMatch"
        )
        self.assertTrue(logged_group["fallbackUsed"])
        self.assertIsNone(logged_group["fallbackReason"])
        self.assertIsNone(logged_group["maximinFallbackReason"])
        self.assertEqual(logged_group["bridgingFallbackReason"], "fewer_than_two_linked")

    def test_text_diagnostics_go_to_the_logger(self):
        service = FakeTextMatchingService()
        with patch.object(
            main, "get_text_matching_service", return_value=service
        ), patch.object(main.log, "log_event") as log_event:
            response = self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {"a": {}, "b": {}, "c": {}},
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()["results"][0]
        for field in ("assignedTarget", "achievedDiversity", "fallbackUsed"):
            self.assertNotIn(field, body)

        call = next(
            call
            for call in log_event.call_args_list
            if call.kwargs.get("extra_data", {}).get("algorithm") == "textGroupMatch"
        )
        extra = call.kwargs["extra_data"]
        logged_group = extra["groups"][0]
        self.assertEqual(logged_group["assignedTarget"], 0.5)
        self.assertEqual(logged_group["achievedDiversity"], 0.62)
        self.assertIs(logged_group["fallbackUsed"], False)
        self.assertEqual(extra["condition_counts"], {"medium": 1})
        self.assertEqual(call.kwargs["request"].url.path, "/match")

    def test_thin_high_arm_is_flagged(self):
        service = FakeTextMatchingService()
        with patch.object(
            main, "get_text_matching_service", return_value=service
        ), patch.object(main.log, "log_event") as log_event:
            self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {"a": {}, "b": {}, "c": {}},
                },
            )

        warnings = [
            call
            for call in log_event.call_args_list
            if call.args and call.args[0] == "WARNING"
        ]
        self.assertTrue(
            any("high" in call.args[1] for call in warnings),
            "a high arm below two groups should be flagged",
        )

    def test_group_size_report_flags_a_plan_mismatch(self):
        ids = [f"p{index}" for index in range(100)]
        groups = [ids[i : i + 5] for i in range(0, 100, 5)]

        report = main._group_size_report(ids, 5, groups)
        self.assertEqual(report["group_count"], 20)
        self.assertEqual(report["planned_group_count"], 20)
        self.assertTrue(report["matches_plan"])
        self.assertTrue(report["all_participants_assigned"])

        dropped = main._group_size_report(ids, 5, groups[:19])
        self.assertFalse(dropped["matches_plan"])
        self.assertFalse(dropped["all_participants_assigned"])
        self.assertEqual(dropped["participants_assigned"], 95)
        self.assertEqual(dropped["missing_participants"], sorted(ids[95:]))

    def test_coverage_is_checked_by_identity_not_by_count(self):
        """Equal counts can hide one participant duplicated and another dropped,
        which is exactly the failure the ERROR path claims to catch."""
        ids = ["a", "b", "c", "d"]
        report = main._group_size_report(ids, 2, [["a", "b"], ["c", "c"]])

        self.assertEqual(report["participants_assigned"], report["participant_count"])
        self.assertFalse(report["all_participants_assigned"])
        self.assertEqual(report["missing_participants"], ["d"])
        self.assertEqual(report["duplicated_participants"], ["c"])

    def test_strict_mode_refuses_placeholder_substitution(self):
        """At a live event, groups built from placeholder text would look
        statistically perfect and mean nothing. REQUIRE_REAL_TEXT turns the
        silent substitution into a listable 422."""
        service = FakeTextMatchingService()
        with patch.object(
            main, "get_text_matching_service", return_value=service
        ), patch.dict(main.os.environ, {"REQUIRE_REAL_TEXT": "1"}):
            response = self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {
                        "a": {"freeTextResponse": "real text"},
                        "b": {},
                        "c": {"freeTextResponse": "   "},
                    },
                },
            )
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["code"], "MISSING_TEXT_RESPONSES")
        self.assertIn("b", body["message"])
        self.assertIn("c", body["message"])
        self.assertIsNone(service.participant_responses)

    def test_strict_mode_off_keeps_placeholder_fallback(self):
        service = FakeTextMatchingService()
        with patch.object(
            main, "get_text_matching_service", return_value=service
        ), patch.dict(main.os.environ, {}, clear=False):
            main.os.environ.pop("REQUIRE_REAL_TEXT", None)
            response = self.client.post(
                "/match",
                json={
                    "algorithm": "textGroupMatch",
                    "targetGroupSize": 3,
                    "participants": {"a": {}, "b": {}, "c": {}},
                },
            )
        self.assertEqual(response.status_code, 200)

    def test_text_algorithm_requires_three_participants(self):
        response = self.client.post(
            "/match",
            json={
                "algorithm": "textGroupMatch",
                "targetGroupSize": 3,
                "participants": {"a": {}, "b": {}},
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "INSUFFICIENT_PARTICIPANTS")

    def test_text_algorithm_requires_target_size_three(self):
        response = self.client.post(
            "/match",
            json={
                "algorithm": "textGroupMatch",
                "targetGroupSize": 2,
                "participants": {"a": {}, "b": {}, "c": {}},
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["code"],
            "TARGET_GROUP_SIZE_TOO_SMALL",
        )


if __name__ == "__main__":
    unittest.main()
