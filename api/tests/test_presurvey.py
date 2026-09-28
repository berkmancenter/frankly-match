import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import presurvey
from presurvey import (
    CATALOG_PATH,
    DIFFUSION_TOPIC_ID,
    PreSurveyDataError,
    link_participants,
    load_comment_catalog,
    normalize_name,
    parse_approval_matrix,
    parse_comment_catalog,
    read_source,
)
from text_match import TextMatchingService


def _catalog_bytes(comments):
    """comments: list of (comment_id, topic_id, author_pid)."""
    return json.dumps({
        "schema_version": 1,
        "topics": {DIFFUSION_TOPIC_ID: {}, "food_access": {}},
        "embedding_metadata": {"model": "m", "revision": "r", "dimensions": 2},
        "comments": {
            cid: {
                "comment_id": cid,
                "text": f"text {cid}",
                "topic_id": topic,
                "author_pid": author,
                "embedding": [1.0, 0.0],
            }
            for cid, topic, author in comments
        },
    }).encode()


CATALOG = parse_comment_catalog(_catalog_bytes([
    ("c1", DIFFUSION_TOPIC_ID, "pid_a"),
    ("c2", DIFFUSION_TOPIC_ID, "pid_b"),
    ("c3", "food_access", "pid_a"),
]))


def _matrix_csv(header, rows):
    lines = [",".join(header)] + [",".join(str(cell) for cell in row) for row in rows]
    return ("\n".join(lines) + "\n").encode()


MATRIX = parse_approval_matrix(
    _matrix_csv(
        ["email", "name", "pid", "c3", "c1", "c2"],
        [
            ["ana@example.org", "José García", "pid_a", 0.25, 1, 0],
            ["bo@example.org", "Bo Lee", "pid_b", 0.5, 0.75, 1],
            ["cy@example.org", "Sam Smith", "pid_c", 0, 0, 0],
            ["dee@example.org", "Sam Smith", "pid_d", 1, 1, 1],
        ],
    ),
    CATALOG,
    "test",
)


class CommittedCatalogTests(unittest.TestCase):
    def test_committed_catalog_loads_and_is_consistent(self):
        catalog = load_comment_catalog()
        self.assertEqual(len(catalog.comments), 289)
        self.assertEqual(
            catalog.topic_counts(),
            {"stocking_growing": 120, "food_access": 91, "food_affordability": 78},
        )
        self.assertEqual(catalog.embeddings.shape, (289, 768))
        self.assertEqual(catalog.sha256, hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest())
        self.assertEqual(catalog.embedding_model, "cartgr/embeddings-for-preferences-st5-xl")

    def test_committed_catalog_carries_no_names_or_emails(self):
        document = json.loads(CATALOG_PATH.read_bytes())
        fields = {key for entry in document["comments"].values() for key in entry}
        self.assertFalse(fields & {"email", "name", "first_name", "last_name", "phone"})


class CatalogValidationTests(unittest.TestCase):
    def test_rejects_embeddings_that_are_not_unit_length(self):
        document = json.loads(_catalog_bytes([("c1", DIFFUSION_TOPIC_ID, "a")]))
        document["comments"]["c1"]["embedding"] = [2.0, 0.0]
        with self.assertRaises(PreSurveyDataError):
            parse_comment_catalog(json.dumps(document).encode())

    def test_rejects_a_catalog_without_the_diffusion_topic(self):
        document = json.loads(_catalog_bytes([("c1", DIFFUSION_TOPIC_ID, "a")]))
        document["topics"] = {"food_access": {}}
        document["comments"]["c1"]["topic_id"] = "food_access"
        with self.assertRaises(PreSurveyDataError):
            parse_comment_catalog(json.dumps(document).encode())


class ApprovalMatrixTests(unittest.TestCase):
    def test_columns_are_reordered_into_catalog_order(self):
        # The CSV lists c3, c1, c2; the catalog order is c1, c2, c3.
        np.testing.assert_array_equal(MATRIX.probabilities[0], [1, 0, 0.25])
        self.assertEqual(MATRIX.pids, ("pid_a", "pid_b", "pid_c", "pid_d"))

    def test_observed_share_counts_exact_zeros_and_ones(self):
        # 12 cells, of which 0.25, 0.5 and 0.75 are the only predictions.
        self.assertAlmostEqual(MATRIX.observed_share, 9 / 12)

    def test_byte_order_mark_is_ignored(self):
        raw = b"\xef\xbb\xbf" + _matrix_csv(
            ["email", "name", "pid", "c1", "c2", "c3"], [["e", "n", "p", 1, 0, 1]]
        )
        self.assertEqual(parse_approval_matrix(raw, CATALOG, "t").emails, ("e",))

    def test_rejects_columns_that_do_not_match_the_catalog(self):
        for header in (
            ["email", "name", "pid", "c1", "c2"],
            ["email", "name", "pid", "c1", "c2", "c3", "c4"],
            ["email", "name", "pid", "c1", "c2", "c2"],
            ["name", "pid", "c1", "c2", "c3"],
        ):
            with self.subTest(header=header), self.assertRaises(PreSurveyDataError):
                row = ["x"] * 3 + [0] * (len(header) - 3)
                parse_approval_matrix(_matrix_csv(header, [row]), CATALOG, "t")

    def test_rejects_values_outside_zero_to_one(self):
        raw = _matrix_csv(["email", "name", "pid", "c1", "c2", "c3"], [["e", "n", "p", 1.5, 0, 0]])
        with self.assertRaises(PreSurveyDataError):
            parse_approval_matrix(raw, CATALOG, "t")

    def test_rejects_duplicate_pids(self):
        raw = _matrix_csv(
            ["email", "name", "pid", "c1", "c2", "c3"],
            [["a", "n", "p", 0, 0, 0], ["b", "m", "p", 0, 0, 0]],
        )
        with self.assertRaises(PreSurveyDataError):
            parse_approval_matrix(raw, CATALOG, "t")

    def test_real_matrix_matches_the_committed_catalog_when_present(self):
        """Runs only where the private matrix sits in the repo root."""
        path = Path(__file__).resolve().parents[2] / "approval_matrix.csv"
        if not path.exists():
            self.skipTest("private approval matrix not present")
        matrix = parse_approval_matrix(path.read_bytes(), load_comment_catalog(), str(path))
        authors = {comment.author_pid for comment in load_comment_catalog().comments}
        self.assertTrue(authors <= set(matrix.pids))


class SourceTests(unittest.TestCase):
    def test_local_paths_are_read_directly(self):
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            handle.write(b"data")
            handle.flush()
            self.assertEqual(read_source(handle.name), b"data")

    def test_gcs_uris_split_into_bucket_and_object(self):
        with patch.object(presurvey, "_download_gcs", return_value=b"x") as download:
            read_source("gs://deliberations-prod/match-api-artifacts/m.csv")
        download.assert_called_once_with("deliberations-prod", "match-api-artifacts/m.csv")
        with self.assertRaises(PreSurveyDataError):
            read_source("gs://bucket-only")


class LinkingTests(unittest.TestCase):
    def test_email_ignores_case_and_surrounding_space(self):
        link = link_participants({"p": ("  ANA@Example.org ", None)}, MATRIX)["p"]
        self.assertEqual((link.method, link.presurvey_pid), ("email", "pid_a"))

    def test_email_wins_over_a_conflicting_name(self):
        link = link_participants({"p": ("bo@example.org", "José García")}, MATRIX)["p"]
        self.assertEqual((link.method, link.presurvey_pid), ("email", "pid_b"))

    def test_name_is_the_fallback_and_ignores_accents(self):
        link = link_participants({"p": ("new@example.org", "jose  GARCIA")}, MATRIX)["p"]
        self.assertEqual((link.method, link.presurvey_pid), ("name", "pid_a"))

    def test_ambiguous_names_are_never_guessed(self):
        link = link_participants({"p": (None, "Sam Smith")}, MATRIX)["p"]
        self.assertEqual((link.method, link.row), ("none", None))

    def test_missing_identity_is_unmatched(self):
        self.assertEqual(link_participants({"p": (None, None)}, MATRIX)["p"].method, "none")

    def test_name_normalisation(self):
        self.assertEqual(normalize_name("  Zoë\tO'Brien "), "zoe o'brien")


class ServicePreSurveyTests(unittest.TestCase):
    """Loading happens after groups are final; nothing here may affect them."""

    def _run(self, uri, identities):
        embeddings = np.random.default_rng(1).normal(size=(9, 4))

        class Client:
            calls = [embeddings, np.ones((1, 4))]

            def embed(self, sentences):
                return self.calls.pop(0)

        service = TextMatchingService(
            embedding_client=Client(),
            optimization_seconds=0, approval_matrix_uri=uri,
        )
        with patch("text_match.log.log_event") as logged:
            groups = service.match({f"p{i}": f"t{i}" for i in range(9)}, 3, identities=identities)
        return groups, logged.call_args_list

    def test_links_are_logged_when_the_matrix_loads(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("text_match.load_comment_catalog", return_value=CATALOG):
            path = Path(folder) / "matrix.csv"
            path.write_bytes(_matrix_csv(
                ["email", "name", "pid", "c1", "c2", "c3"],
                [["ana@example.org", "Ana", "pid_a", 1, 0, 1]],
            ))
            identities = {f"p{i}": (None, None) for i in range(9)}
            identities["p0"] = ("Ana@example.org", "Ana")
            groups, calls = self._run(str(path), identities)

        record = next(
            call.kwargs["extra_data"] for call in calls
            if call.args[1] == "Pre-survey linking"
        )
        self.assertEqual(record["link_counts"], {"email": 1, "name": 0, "none": 8})
        linked = next(entry for entry in record["links"] if entry["participant_id"] == "p0")
        self.assertEqual(linked["presurvey_pid"], "pid_a")
        self.assertEqual(sum(len(g.participant_ids) for g in groups), 9)

    def test_a_failed_matrix_load_is_logged_and_leaves_groups_intact(self):
        baseline, _ = self._run(None, {})
        with patch("text_match.load_approval_matrix", side_effect=OSError("denied")):
            groups, calls = self._run("gs://bucket/m.csv", {"p0": ("a@b.c", None)})

        self.assertEqual(
            [g.participant_ids for g in groups], [g.participant_ids for g in baseline]
        )
        data = next(
            call.kwargs["extra_data"] for call in calls if call.args[1] == "Pre-survey data"
        )
        self.assertFalse(data["matrix_loaded"])
        self.assertIn("denied", data["matrix_error"])
        self.assertTrue(any(
            call.args[0] == "WARNING" and "Approval matrix unavailable" in call.args[1]
            for call in calls
        ))

    def test_a_missing_uri_is_a_warning_not_an_error(self):
        _, calls = self._run(None, {})
        warning = next(
            call for call in calls if "Approval matrix unavailable" in call.args[1]
        )
        self.assertEqual(warning.args[0], "WARNING")
        self.assertIn("APPROVAL_MATRIX_URI", warning.args[1])


if __name__ == "__main__":
    unittest.main()
