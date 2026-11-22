# Results: `single`

- Date: 2026-11-22 (UTC)
- Model: `gemini-3.1-flash-lite` (evaluation mode, D4)
- Prompt version: `aa94f1ae2cf6` (`src/reviewer/prompts/single.md`, SHA-256, first 12 hex)
- Limits: `max_model_calls` 25, `max_tool_calls` 40 (`config/review.yaml` `limits.single`)
- Items: 20 (6 clean); line tolerance 3

## Recall and precision

| Mode | Bucket | Items | Recall | Precision | tp | fp | fn | Discarded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict | all | 20 | 0.07 | 0.50 | 2 | 2 | 26 | 0 |
| strict | small | 6 | 0.00 | 0.00 | 0 | 1 | 5 | 0 |
| strict | medium | 4 | 0.00 | 0.00 | 0 | 1 | 6 | 0 |
| strict | large | 10 | 0.12 | 1.00 | 2 | 0 | 15 | 0 |
| location_only | all | 20 | 0.14 | 1.00 | 4 | 0 | 24 | 0 |
| location_only | small | 6 | 0.20 | 1.00 | 1 | 0 | 4 | 0 |
| location_only | medium | 4 | 0.17 | 1.00 | 1 | 0 | 5 | 0 |
| location_only | large | 10 | 0.12 | 1.00 | 2 | 0 | 15 | 0 |

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
| item_006 | large | complete | 0 | 0 | 0 | 2 | 0 | 0 | evidence 1 | 13 | 12 | 81780 / 673 | 20.7 | 0.0 |
| item_007 | large | complete | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 7 | 6 | 59197 / 273 | 10.5 | 39.8 |
| item_008 | large | complete | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 8 | 7 | 52696 / 293 | 14.0 | 86.6 |
| item_009 | large | complete | 0 | 0 | 0 | 0 | 0 | 0 | evidence 1 | 8 | 7 | 101717 / 487 | 16.5 | 0.0 |
| item_010 | medium | complete | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 9 | 8 | 106879 / 288 | 15.4 | 36.3 |
| item_011 | small | complete | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 7 | 6 | 55078 / 253 | 11.6 | 1.6 |
| item_012 | small | complete | 0 | 0 | 0 | 0 | 0 | 0 | evidence 1 | 4 | 3 | 14816 / 440 | 6.6 | 35.8 |
| item_013 | large | complete | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 4 | 3 | 15185 / 309 | 6.3 | 1.3 |
| item_014 | large | complete | 0 | 0 | 0 | 3 | 0 | 0 | evidence 2 | 7 | 6 | 68241 / 646 | 12.5 | 36.3 |
| item_015 | large | complete | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 10 | 9 | 56721 / 285 | 22.0 | 1.8 |
| item_016 | large | complete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 15 | 174240 / 523 | 31.0 | 63.8 |
| item_017 | medium | complete | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 6 | 5 | 21899 / 257 | 9.7 | 3.2 |
| item_018 | medium | complete | 0 | 0 | 0 | 0 | 0 | 0 | evidence 1 | 10 | 9 | 96113 / 893 | 15.6 | 40.1 |
| item_019 | small | complete | 1 | 0 | 1 | 2 | 1 | 0 | 0 | 7 | 6 | 38843 / 395 | 10.8 | 3.3 |
| item_020 | small | complete | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 6 | 5 | 24220 / 231 | 8.4 | 34.9 |

## Cost and time

- Median review time with rate-limit waiting removed (D7): 14.3 s; median wait beside it: 35.3 s
- Mean tokens per review: 75138 input, 415 output
- Estimated cost per review: $0.01941 at $0.25 / $1.50 per million input / output tokens (D5 estimate; the free tier was used, so nothing was billed)

## Dropped findings (post-processing, decision 0007)

| Reason | Count |
| --- | --- |
| schema | 0 |
| not_in_diff | 0 |
| evidence | 6 |
| duplicate | 0 |

## Scoring decisions (decision 0010)

- Items scored: 15; items needing a re-run: 0
- Recorded runs: 20, of which `rerun_required` (not scored, re-run): 0

| Decision | Cause | Items |
| --- | --- | --- |
| scored | complete | 15 |
