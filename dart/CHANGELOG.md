## 0.2.0

**Breaking change**: all three public functions now return `List<MatchGroup>` instead of `List<List<String>>`.

`MatchGroup` is a new class with two fields:
- `id` — string identifier for the group within a single match call
- `participantIds` — the list of participant IDs (previously the inner list)

Migration: replace uses of the return value as `List<List<String>>` with `.map((g) => g.participantIds)` where needed, and use `g.id` to access the group identifier.

This change aligns the Dart package's return shape with the HTTP API (`groupId` + `participantIds`), allowing consuming apps to treat both interchangeably.

## 0.1.1
Update README to include Usage section.

## 0.1.0

Initial pub.dev release of the Frankly Match diversity matching algorithm!

Publishes three functions:
- `bucketMatch` — pair matching via direct Hamming distance maximization
- `groupMatch` — group matching via BFS cluster-then-compose
- `randomGroups` — random assignment baseline

Previously this code lived as a path dependency inside the Frankly monorepo (`firebase/functions`).