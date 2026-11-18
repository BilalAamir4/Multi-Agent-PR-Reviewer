# 0002: Scoring details for D9 to D9b

Date: Nov 19, 2026 · Status: accepted · Implements: D8, D8a, D9, D9a, D9b in `eval/score.py`

These are the interpretations made in stage 0A to turn D9 to D9b into code. `eval/score.py` must
not change after the first evaluation run without a new decision record.

- **Changed lines.** Target (new-file) side only, per non-deleted file, keyed by unidiff's
  `patched_file.path` (no `a/` or `b/`; renames use the new name). An added line counts at its
  target line. A removed line counts at the target line of the next line in the same hunk that
  has one; if none follows, at the hunk's last target line, `max(target_start, target_start +
  target_length - 1)`.
- **Discarded findings.** A finding is in-diff when some line of its range is within
  `LINE_TOLERANCE` of a changed line in the same file. Other findings are discarded: counted and
  listed, but excluded from every precision, recall, false-alarm and adjusted-precision number.
- **Matching.** One-to-one maximum bipartite matching (augmenting paths), separately for strict
  and location-only. Bugs are tried in `bug_id` order and findings in index order, so results are
  deterministic and the number of hits never depends on finding order.
- **Overlap.** `f.line_start <= b.line_end + 3 and f.line_end >= b.line_start - 3`, same file.
  Strict also needs `category != "other"` and `category in accepted_kinds`.
- **Duplicates.** Unmatched in-diff findings that could have matched an already-matched bug.
  They are already counted in `fp` (D9b) and also reported separately for information.
- **Finding IDs.** `"{item_id}#{index}"`, index = 0-based position in that item's findings file.
- **Adjusted precision** (strict only). `(tp + R) / (tp + fp)`, where R = strict-unmatched in-diff
  findings labelled `real_bug` in `manual_labels.json`. Unlabelled ones count as false alarms and
  are reported as `unlabelled_unmatched`. No labels file: `None`. A label for an unknown finding ID
  is an error. Labels on discarded or matched findings are allowed and ignored.
- **Size buckets.** By changed (added or modified, not deleted) `.py` files: small 0 to 3, medium
  4 to 7, large 8 or more.
- **LINE_TOLERANCE = 3** is a module constant in `eval/score.py`, not config, so it cannot vary
  between runs.
- **Ratios with an empty denominator** are `None` (recall with no planted bugs, precision with no
  in-diff findings, false alarms per clean PR with no clean items).
- **Input checks.** Findings files are `<results>/<config>/item_*.json`; each file's `item_id`
  must equal its file name and its `config` must equal the folder name. Missing files, files for
  items not in the dataset, and an empty dataset are errors (CLI exit code 2).
