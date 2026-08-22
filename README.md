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

Text matching prioritizes group size before diversity. Every group contains at least three participants, and the size planner first maximizes the number of groups exactly equal to `targetGroupSize`. The matcher then estimates the achievable minimum and maximum average pairwise cosine distance for each group size.

Each group is then assigned one of three targets — low, medium, or high — as a raw cosine distance rather than a normalized value, so targets mean the same thing across events. Group formation can only redistribute the disagreement already present in the pool, so achieved diversities average to roughly the pool mean and targets have to be set within that limit: low groups target the achievable minimum, medium groups the pool mean, and the high target is whatever the remaining budget allows, capped at the achievable maximum. Roughly a quarter of groups are low and a tenth are high. Events with fewer than three groups cannot support three levels, so every group targets the pool mean. A time-bounded neighborhood search then moves each group toward its target without changing its size.

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

Text results add `diversityLevel`, `assignedTarget`, `achievedDiversity`, `diffusionStatement`, and `fallbackUsed` to the existing `groupId` and `participantIds` fields.

Text responses also return a top-level `participantResponses` map giving the text that was actually embedded for each participant. Persist this alongside the groups: participants who supply no `freeTextResponse` receive a deterministic placeholder, so it is the only record of what the groups were built from, and re-embedding it reproduces the distance matrix that downstream analysis needs.

## Tests

```bash
cd api
python -m unittest discover -s tests
```

Tests mock the embedding endpoint. They do not send participant text or credentials to an external service.
