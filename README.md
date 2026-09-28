# Frankly Match

Frankly Match is a research and engineering effort to develop effective ways to match people into groups for constructive dialogue.

## Contents

- `dart/` — [`frankly_match`](https://pub.dev/packages/frankly_match) package containing the original local matching algorithms.
- `api/` — FastAPI service deployed on Google Cloud Run.
- `demo/` — Static browser demonstration of the hosted API.

## Algorithms

The Dart package exposes:

- **`bucketMatch`** — directly maximizes pairwise Hamming distance.
- **`groupMatch`** — clusters similar binary answer masks, then composes diverse groups.
- **`randomGroups`** — random assignment baseline.

The HTTP API selects an algorithm through the request's `algorithm` field:

- **`binaryGroupMatch`** — the existing binary-mask group matcher.
- **`textGroupMatch`** — embeds free-form responses and matches groups toward numeric diversity targets.

Text matching runs in five stages.

**1. Plan.** `plan_group_sizes` maximizes the number of groups exactly equal to
`targetGroupSize`, never smaller than three. Groups are then allocated to arms as
`sqrt(2) : 1 : 1` (medium : low : high). Off-size groups go to the medium arm, so
the extreme arms — which carry the contrast — stay uniform and have a single
well-defined feasible bound each.

**2. Randomize.** Participants are shuffled and split into three pools, one per
arm. Assignment to condition is therefore exactly random; the optimizer only
rearranges people *within* their own pool afterwards and can never decide who
experiences which condition. This happens after embedding, so a
`REQUIRE_REAL_TEXT` failure aborts before anyone is assigned.

**3. Optimize the endpoints, minimax.** The low pool minimizes its *worst*
group's diversity; the high pool maximizes its worst. Minimax rather than
mean-squared error because an arm label is only meaningful if it holds for every
group in the arm, not on average.

**4. Place the medium arm.** The medium target is the midpoint of the two
*achieved* endpoint means, so the three doses are equally spaced — which is what
preserves the identity `beta_2 = -C/2` that the confirmatory contrast relies on.
The medium pool is then optimized toward that value by mean-squared error.

**5. Validate.** Arms are checked for overlap, and each arm reports whether its
independent restarts agreed. Agreement is the acceptance criterion: the feasible
floor and ceiling describe what a *single* optimally-chosen group can reach,
which minimax cannot match when every group in the arm must clear the bar
simultaneously. Restarts converging on the same value is the evidence that an arm
sits at its pool's real limit rather than being stuck.

Each text-matched group also receives a reusable `diffusionStatement`. The selected statement maximizes its minimum cosine distance from any member of the group.

## Text Response Transition

A participant sends either `freeTextResponses`, a map from question id to answer, or a single `freeTextResponse`. Answers are only compared with answers to the same question: `textGroupMatch` computes a cosine distance per question and averages it over the questions both people answered. Averaging distances, rather than embeddings, stops opposite disagreements on different topics from cancelling out. A pair with no question in common gets the mean distance of the other pairs, which leaves the pool mean unchanged.

Neither field is yet guaranteed by the upstream survey payload. During this transition, participants with no text receive deterministic development placeholders. The replacement point is marked with a TODO in `api/main.py`.

## Local API

Create a local environment file:

```bash
cp .env.example .env
```

Set `HF_TOKEN` in `.env`, then run:

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The main endpoint is `POST /match`, and interactive documentation is available at `http://127.0.0.1:8000/docs`.

Example text request:

```json
{
  "algorithm": "textGroupMatch",
  "targetGroupSize": 5,
  "participants": {
    "alice": {
      "freeTextResponse": "Public transit should be free in major cities."
    },
    "bob": {
      "freeTextResponse": "Transit fares help maintain reliable service."
    },
    "carol": {
      "freeTextResponse": "Cities should invest more heavily in rail."
    },
    "dave": {
      "freeTextResponse": "Neighborhoods should control new development."
    },
    "eve": {
      "freeTextResponse": "Dense housing makes cities more affordable."
    }
  }
}
```

Text results add `diversityLevel` and `diffusionStatement` to the existing `groupId` and `participantIds` fields.

The matching diagnostics are no longer in the response. `assignedTarget`,
`achievedDiversity`, `fallbackUsed` and the text that was actually embedded all
go to Google Cloud Logging instead (see Logging below).

## Logging

Every `/match` call emits structured entries through `api/logger.py`:

- the embedded text per participant, flagged where a placeholder was
  substituted.
- the resulting groups with assigned target, achieved diversity, diffusion
  statement and fallback flag.
- a group-size report comparing produced groups against `plan_group_sizes`,
  including `missing_participants`, `duplicated_participants` and
  `unexpected_participants` so coverage is checked by identity rather than by
  count, plus `condition_counts` giving the number of groups in each arm.
- an `arms` payload, one entry per diversity arm: `level`, `groups`, `people`,
  `sizes`, `target` (medium only), `achieved` and `achieved_mean`, the
  achievable `floor`/`ceiling`/`margin` for that pool, and the optimizer's
  status as `restarts_used`, `restart_statistics`, `restart_spread`,
  `converged` and `deadline_bound`.
- per-event geometry: `pool_mean`, the achieved `endpoint_low` and
  `endpoint_high`, the derived `medium_target`, and `arms_separated` with the
  `low_to_medium_gap` and `medium_to_high_gap`.
- doses on the calibrated Bradley-Terry axis as `achieved_mean_bt`,
  `arccos(1 - d) / pi` -- the predicted fraction of voters who would split on a
  pair at distance `d`.

Groups are allocated to diversity arms as `sqrt(2) : 1 : 1`
(medium : low : high) -- 6 / 8 / 6 at 20 groups, 7 / 11 / 7 at 25. This is a
compromise, not a single optimum: the primary curvature contrast `2M - L - H`
alone would want `2 : 1 : 1`, the pairwise `M - H` comparison wants
`sqrt(2) : 1 : 1`, and the linear contrast `H - L` wants fatter extreme arms.
Relative to allocating purely for the primary, this costs about 1.7 points of
power on the curvature test and returns about 5.3 on the linear one. Keeping
the extreme arms equal makes the linear and quadratic contrasts exactly
orthogonal.

Size mismatches and unassigned participants raise `WARNING` and `ERROR`. A high
arm below two groups raises a `WARNING`, since the contrast is not estimable.

**Retention.** These logs are now the system of record for the embedded text.
The default Cloud Logging bucket expires entries after 30 days, which is shorter
than any study timeline — route a sink to BigQuery or GCS before relying on this.

### Study linkage and reproducibility

Send optional `studyId` and `eventId` strings in the `/match` request. Every
request receives a new `match_run_id`, returned in the `X-Match-Run-ID` response
header (also exposed to browser clients) on every response, including 400/422
validation failures and 500 errors, so failed attempts and retries can be told
apart. Unhandled errors return `{"code": "INTERNAL_ERROR"}` with the traceback
logged under `Match request failed`. Save that header alongside the groups
actually used. Group IDs restart at `1`, so the group key is
`(match_run_id, groupId)`. Retries receive distinct run IDs. Response bodies and
matching behavior are unchanged. All logs within the matching request, including
validation failures and embedding retries, carry the run ID and
`log_schema_version: 1`; matching records additionally carry `study_id` and
`event_id`. Completion records include HTTP status and elapsed seconds.

`Matching configuration` records the ordered participant IDs, seed as a decimal
string (to avoid loss of 64-bit precision), seed method, group sizes, optimization
budget, restart count, allocation weight, metric definition, and candidate
discussion statements. Set `MATCH_CODE_REVISION`, `HF_MODEL_ID`, and
`HF_MODEL_REVISION` at deployment to record provenance: the full git commit SHA
of the deployed code, the model repository the endpoint serves (`owner/name`),
and that repository's commit SHA. Unset or blank values are logged as null; the
API cannot infer which model revision a hosted endpoint serves. The seed
reproduces pool assignment, but the wall-clock optimization budget means reruns
need not produce identical final groups.

`Participant distance rows` records the exact cosine-distance matrix used for
matching. Rows follow `participant_order`. Each entry carries a block of whole
rows starting at `row_start` (`row_count` of them, with their `participant_ids`),
packed so that an entry stays under the Cloud Logging size limit: about 6,000
distances per entry, so 100 participants need 2 entries and 500 need 42.
`Participant distances complete` records the matrix size, embedding dimensions,
`entry_count` (the number of row entries to expect), and the SHA-256 of the
matrix's row-major little-endian float64 bytes, so a reconstruction can be
checked without re-embedding text against a possibly changed model. Check that
`entry_count` entries are present and the checksum matches before analysis.

Each entry is a synchronous write on the request path, so events above
`DISTANCE_LOG_MAX_PARTICIPANTS` (500) skip the row export: a `WARNING` is
logged, `rows_logged` is false on the completion record, and only the checksum
remains. A durable per-run export (for example one GCS object per run ID) is
the right long-term home for this matrix and is not part of this change.

Group logs distinguish `participant_embedding_failed` from
`statement_embedding_failed` in `fallbackReason`. Successful matching has a null
reason. On Cloud Logging failure, the same structured payload is serialized as
JSON in the stderr log message, and the failure diagnostic itself is a
structured record carrying the same run ID and schema version. Logging remains
best-effort and is not a durable archive or a transaction. A completion marker alone does not prove all writes
succeeded. Verify the study export before relying on it.

For survey analysis, join outcomes using participant ID and the matching run
actually used, retaining randomized `diversityLevel` separately from measured
`achievedDiversity`. Attendance, final deliberation membership, and survey
completion must be recorded by the calling platform; this API observes only
planned groups. Enable `REQUIRE_REAL_TEXT=1` for study deployments.

## Tests

```bash
cd api
python -m unittest discover -s tests
```

Tests mock the embedding endpoint. They do not send participant text or credentials to an external service.
