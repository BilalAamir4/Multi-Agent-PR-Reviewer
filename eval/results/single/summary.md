# Results: `single`

- Date: 2026-11-20 (UTC)
- Model: `gemini-3.1-flash-lite` (evaluation mode, D4)
- Prompt version: `aa94f1ae2cf6` (`src/reviewer/prompts/single.md`, SHA-256, first 12 hex)
- Limits: `max_model_calls` 25, `max_tool_calls` 40 (`config/review.yaml` `limits.single`)
- Items: 5 (2 clean); line tolerance 3

## Recall and precision

| Mode | Bucket | Items | Recall | Precision | tp | fp | fn | Discarded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict | all | 5 | 0.17 | 0.50 | 1 | 1 | 5 | 0 |
| strict | small | 2 | 0.00 | - | 0 | 0 | 1 | 0 |
| strict | medium | 1 | 0.00 | 0.00 | 0 | 1 | 2 | 0 |
| strict | large | 2 | 0.33 | 1.00 | 1 | 0 | 2 | 0 |
| location_only | all | 5 | 0.33 | 1.00 | 2 | 0 | 4 | 0 |
| location_only | small | 2 | 0.00 | - | 0 | 0 | 1 | 0 |
| location_only | medium | 1 | 0.50 | 1.00 | 1 | 0 | 1 | 0 |
| location_only | large | 2 | 0.33 | 1.00 | 1 | 0 | 2 | 0 |

- False alarms per clean PR: strict 0.00, location-only 0.00
- Adjusted precision (D9b): not computed; no manual labels yet.

## Per item

| Item | Bucket | Status | Findings | tp | fp | fn | Loc tp | Discarded | Dropped | Model calls | Tool calls | Tokens in / out | Net s | Wait s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item_001 | small | complete | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 10 | 9 | 105395 / 343 | 17.0 | 0.0 |
| item_002 | small | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 5 | 33942 / 240 | 10.2 | 39.1 |
| item_003 | medium | complete | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 22 | 21 | 293901 / 739 | 40.7 | 40.9 |
| item_004 | large | complete | 1 | 1 | 0 | 2 | 1 | 0 | 0 | 7 | 6 | 61775 / 447 | 15.1 | 28.5 |
| item_005 | large | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 8 | 40124 / 293 | 14.6 | 37.5 |

## Cost and time

- Median review time with rate-limit waiting removed (D7): 15.1 s; median wait beside it: 37.5 s
- Mean tokens per review: 107027 input, 412 output
- Estimated cost per review: $0.02738 at $0.25 / $1.50 per million input / output tokens (D5 estimate; the free tier was used, so nothing was billed)

## Dropped findings (post-processing, decision 0007)

| Reason | Count |
| --- | --- |
| schema | 0 |
| not_in_diff | 0 |
| evidence | 0 |
| duplicate | 0 |
