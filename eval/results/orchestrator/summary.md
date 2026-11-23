# Results: `orchestrator`

- Date: 2026-11-23 (UTC)
- Model: `gemini-3.1-flash-lite` (evaluation mode, D4)
- Prompt version: `4eda79561a8c` (`src/reviewer/prompts/orchestrator.md`, SHA-256, first 12 hex)
- Limits: `max_model_calls` 8, `max_tool_calls` 12 (`config/review.yaml` `limits.orchestrator`)
- Items: 20 (6 clean); line tolerance 3

## Recall and precision

| Mode | Bucket | Items | Recall | Precision | tp | fp | fn | Discarded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict | all | 20 | 0.36 | 0.67 | 10 | 5 | 18 | 0 |
| strict | small | 6 | 0.80 | 1.00 | 4 | 0 | 1 | 0 |
| strict | medium | 4 | 0.17 | 0.50 | 1 | 1 | 5 | 0 |
| strict | large | 10 | 0.29 | 0.56 | 5 | 4 | 12 | 0 |
| location_only | all | 20 | 0.43 | 0.80 | 12 | 3 | 16 | 0 |
| location_only | small | 6 | 0.80 | 1.00 | 4 | 0 | 1 | 0 |
| location_only | medium | 4 | 0.33 | 1.00 | 2 | 0 | 4 | 0 |
| location_only | large | 10 | 0.35 | 0.67 | 6 | 3 | 11 | 0 |

- False alarms per clean PR: strict 0.17, location-only 0.17
- Adjusted precision (D9b): not computed; no manual labels yet.

## Per item

| Item | Bucket | Status | Findings | tp | fp | fn | Loc tp | Discarded | Dropped | Model calls | Tool calls | Tokens in / out | Net s | Wait s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item_001 | small | complete | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 20 | 17 | 132832 / 1212 | 36.5 | 43.2 |
| item_002 | small | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 12 | 53420 / 735 | 24.6 | 39.1 |
| item_003 | medium | partial | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 39 | 38 | 222545 / 1632 | 30.4 | 348.3 |
| item_004 | large | complete | 1 | 1 | 0 | 2 | 1 | 0 | evidence 1 | 20 | 17 | 170247 / 1257 | 47.2 | 54.9 |
| item_005 | large | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 29 | 255168 / 1286 | 53.9 | 89.8 |
| item_006 | large | complete | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 22 | 19 | 119504 / 1237 | 35.9 | 68.2 |
| item_007 | large | partial | 1 | 0 | 1 | 3 | 1 | 0 | 0 | 30 | 29 | 329672 / 1484 | 27.5 | 145.8 |
| item_008 | large | limit_reached | 1 | 0 | 1 | 2 | 0 | 0 | 0 | 26 | 24 | 186710 / 1280 | 39.2 | 336.0 |
| item_009 | large | complete | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 30 | 26 | 304738 / 1530 | 44.0 | 86.1 |
| item_010 | medium | complete | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 4 | 3 | 13481 / 268 | 5.8 | 39.3 |
| item_011 | small | complete | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 29 | 25 | 141142 / 1448 | 130.0 | 86.2 |
| item_012 | small | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 27 | 24 | 129542 / 1196 | 37.1 | 84.8 |
| item_013 | large | complete | 2 | 1 | 1 | 1 | 1 | 0 | 0 | 27 | 23 | 174670 / 1475 | 49.2 | 90.9 |
| item_014 | large | complete | 2 | 2 | 0 | 1 | 2 | 0 | duplicate 1 | 26 | 22 | 168855 / 1497 | 55.4 | 67.1 |
| item_015 | large | complete | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 31 | 28 | 237207 / 1217 | 50.3 | 100.0 |
| item_016 | large | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 25 | 215363 / 1061 | 52.1 | 97.9 |
| item_017 | medium | limit_reached | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 30 | 28 | 194196 / 1511 | 98.5 | 49.2 |
| item_018 | medium | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 15 | 85641 / 881 | 26.3 | 105.3 |
| item_019 | small | complete | 1 | 1 | 0 | 1 | 1 | 0 | evidence 2 | 22 | 20 | 97048 / 1685 | 225.3 | 89.6 |
| item_020 | small | complete | 1 | 1 | 0 | 0 | 1 | 0 | duplicate 1 | 15 | 12 | 53970 / 1097 | 66.9 | 25.0 |

## Cost and time

- Median review time with rate-limit waiting removed (D7): 45.6 s; median wait beside it: 86.2 s; for the orchestrator the removed wait is the wall-clock union of all waits (wait_s_wall, decision 0009), median 76.5 s
- Mean tokens per review: 164298 input, 1249 output
- Estimated cost per review: $0.04295 at $0.25 / $1.50 per million input / output tokens (D5 estimate; the free tier was used, so nothing was billed)

## Dropped findings (post-processing, decision 0007)

| Reason | Count |
| --- | --- |
| schema | 0 |
| not_in_diff | 0 |
| evidence | 3 |
| duplicate | 2 |

## Scoring decisions (decision 0010)

- Items scored: 20; items needing a re-run: 0
- Recorded runs: 20, of which `rerun_required` (not scored, re-run): 0

| Decision | Cause | Items |
| --- | --- | --- |
| scored | complete | 16 |
| scored | limit_reached | 2 |
| scored | partial: bug_reviewer limit_reached | 1 |
| scored | partial: security_reviewer limit_reached | 1 |

Spawning: off (forced for this configuration, decision 0012; row 2).
