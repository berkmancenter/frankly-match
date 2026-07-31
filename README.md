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

Events with 3–39 groups use normalized targets `0`, `0.5`, and `1`. Events with 40 or more groups use `0`, `0.25`, `0.5`, `0.75`, and `1`; one group receives `0.5`, while two groups receive `0` and `1`. A time-bounded neighborhood search moves each group toward its assigned target without changing its size. The API returns the assigned target together with raw and normalized achieved diversity.

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

Text results add `assignedTarget`, `achievedDiversity`, `normalizedAchievedDiversity`, `diversityLevel`, `diffusionStatement`, and `fallbackUsed` to the existing `groupId` and `participantIds` fields.

## Tests

```bash
cd api
python -m unittest discover -s tests
```

Tests mock the embedding endpoint. They do not send participant text or credentials to an external service.
