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

`freeTextResponse` is defined in the API contract but is not yet guaranteed by the upstream survey payload. During this transition, missing text responses receive deterministic development placeholders. The replacement point is marked with a TODO in `api/main.py`.

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
  substituted. This is the only record of what the groups were built from, and
  re-embedding it reproduces the distance matrix downstream analysis needs.
- the resulting groups with assigned target, achieved diversity, diffusion
  statement and fallback flag.
- a group-size report comparing produced groups against `plan_group_sizes`, and
  `condition_counts` giving the number of groups in each diversity arm.
- the pool geometry each event's targets were derived from: pool mean, the
  achievable floor and ceiling per group size, and `r_star_by_size`, the
  low-to-high ratio at which the achievable ceiling starts binding.

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

## Tests

```bash
cd api
python -m unittest discover -s tests
```

Tests mock the embedding endpoint. They do not send participant text or credentials to an external service.
