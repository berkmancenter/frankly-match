# Contributing

Thanks for your interest! Here's what you need to know to get oriented and contribute effectively.

## Contributing to the Frankly Match API 
### Repo structure overview

```
api/match.py   # matching logic file, referenced in main.py
api/main.py    # FastAPI wrapper — routing, validation, request/response models
api/openapi.yaml  # API contract, also used as the Google API Gateway config
dart/          # Dart pub.dev package (port of the matching logic)
demo/          # Static demo page
```

The core logic lives in `match.py`. You can add additional files with additional matching logic if needed. `main.py` essentially handles all the API-related code required to make your Python logic into an API.

### Adding a new algorithm

The `/match` endpoint accepts an `algorithm` parameter — currently only `binaryGroupMatch` is supported. Here's how to add another:

**1. Write the algorithm in `match.py`**

You can model your new algorithm after `group_match`. For example:

```python
def new_algorithm(participant_responses: dict[str, str], target_group_size: int) -> list[list[str]]:
    # your logic here
    ...
```

**2. Register your new algorithm in `main.py`**

The `match` route checks for which algorithm it should use within a simple if/else branch. Add a branch for your algorithm:

```python
@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    samples = _normalize_masks(req.participants)
    if req.algorithm == "binaryGroupMatch":
        groups = group_match(samples, req.targetGroupSize)
    elif req.algorithm == "newAlgorithm":
        groups = my_match(samples, req.targetGroupSize)
    return MatchResponse(results=[
        GroupResult(groupId=str(i + 1), participantIds=g)
        for i, g in enumerate(groups)
    ])
```

Also update the `check_algorithm` validator to accept the new name.

**3. Update `openapi.yaml`**

Add the new algorithm name to the `MatchAlgorithm` enum and document what it does.

**4. If your algorithm uses different participant data fields**, add them to `ParticipantData` in `main.py` and document them in `openapi.yaml` under `ParticipantData`. 

## Other contributions

- Bug fixes, tests, and documentation improvements are very welcome
- If you have ideas for new matching heuristics (similarity-maximizing, text-based, etc.), definitely open an issue so the team can discuss!
- The Dart package in `dart/` is separate from the API and has different standards for contributing. The API is intended to be much more experimental.

### Using the demo

`demo/index.html` contains the code for our simple webapp demo, which is hosted on GitHub pages and linked in the repo. This demo hits the real hosted API, so you can open it in a browser and play around with it. You can also check the console to see live response payloads from our hosted API.