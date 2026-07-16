# Frankly Match

Frankly Match is an ongoing research and engineering effort to develop the most effective ways to match people into groups for facilitating constructive dialogue.

Code for the "smart matching" feature that is currently used by the Frankly platform lives here. We welcome contributions, forks, comments, issues, etc. from the community. Thanks for stopping by!

## Contents

- `dart/` — [`frankly_match`](https://pub.dev/packages/frankly_match) pub.dev Dart package.
- `api/` — Google Cloud Run service implementing the [match API](./openapi.yaml). Written in Python for research purposes.

## Description

Frankly Match currently consists of one main algorithm which is based on maximizing [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) between participant survey answers. 

Each participant's survey answers are encoded as a binary string (`"01001"` = one bit per question). The endpoints available are as follows:

- **`bucketMatch`** — for pairs. Directly maximises pairwise Hamming distance.
- **`groupMatch`** — for groups of 3+. BFS-clusters by similarity, then composes groups by taking one member from each cluster.
- **`randomGroups`** — no survey data needed. Random assignment baseline.

We aim to develop other matching algorithms which may optimize for different heuristics (e.g. maximizing similarity, or creating other interesting configurations). We are also interested in expanding the diversity-matching algorithm to allow it to ingest and factor in more than simple binary answers (e.g. text, multiple-choice, drawings?!). 
