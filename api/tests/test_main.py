import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from text_match import TextMatchGroup


class FakeTextMatchingService:
    def __init__(self):
        self.participant_responses = None
        self.target_group_size = None

    def match(self, participant_responses, target_group_size, request=None):
        self.request = request
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
    def setUp(self):
        self.client = TestClient(main.app)

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
        report = main._group_size_report(100, 5, [5] * 20)
        self.assertEqual(report["group_count"], 20)
        self.assertEqual(report["planned_group_count"], 20)
        self.assertTrue(report["matches_plan"])
        self.assertTrue(report["all_participants_assigned"])

        dropped = main._group_size_report(100, 5, [5] * 19)
        self.assertFalse(dropped["matches_plan"])
        self.assertFalse(dropped["all_participants_assigned"])
        self.assertEqual(dropped["participants_assigned"], 95)

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
