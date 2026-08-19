import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from text_match import TextMatchGroup


class FakeTextMatchingService:
    def __init__(self):
        self.participant_responses = None
        self.target_group_size = None

    def match(self, participant_responses, target_group_size):
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
                normalized_achieved_diversity=0.54,
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

    def test_binary_mask_padding_is_deterministic(self):
        participants = {
            "a": main.ParticipantData(binaryAnswerMask="1"),
            "b": main.ParticipantData(binaryAnswerMask="1011010110"),
            "c": main.ParticipantData(binaryAnswerMask="01"),
        }

        first = main._normalize_masks(participants)
        second = main._normalize_masks(participants)

        self.assertEqual(first, second)

    def test_binary_match_is_deterministic_with_unequal_masks(self):
        payload = {
            "algorithm": "binaryGroupMatch",
            "targetGroupSize": 2,
            "participants": {
                "a": {"binaryAnswerMask": "1"},
                "b": {"binaryAnswerMask": "1011010110"},
                "c": {"binaryAnswerMask": "01"},
                "d": {"binaryAnswerMask": "0110"},
            },
        }

        first = self.client.post("/match", json=payload)
        second = self.client.post("/match", json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())

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
                "assignedTarget": 0.5,
                "achievedDiversity": 0.62,
                "normalizedAchievedDiversity": 0.54,
                "diffusionStatement": "A test statement",
                "fallbackUsed": False,
            },
        )

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
