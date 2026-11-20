# 0005: Per-bug extra accepted kinds

Date: Nov 20, 2026 · Status: accepted · Refines D8a (does not replace it)

## Context

D8a puts equivalent kinds into the ground truth before the first run. Global groups live in
`eval/equivalences.yaml` and apply to every bug of a planted kind. Some single bugs can honestly
be described by a second kind without that being true of the kind in general.

## Decision

- **Global groups (D8a) stay limited to `eval/equivalences.yaml`.** Its only group is
  `missing_none_check: [unhandled_exception]`.
- **Per-bug extras.** A recipe bug in `eval/plants/<item_id>.yaml` may set
  `extra_accepted_kinds: [<kinds>]`, which requires `extra_accepted_reason: "<one sentence>"`.
  `eval/build_dataset.py` writes `accepted_kinds = default_accepted_kinds(kind) +
  extra_accepted_kinds`, without duplicates and in that order. `other` is never allowed (it is not
  a plant kind, so the recipe schema rejects it).
- **Every per-bug extra must be listed in the table below**, with the same reason as the recipe.
  `eval/check_dataset.py` check 1 verifies `accepted_kinds`, and check 12 ("extra kinds
  documented") fails for a bug whose extra kinds are not all named in its row here.
- Extras are fixed with the rest of the ground truth from the stage 1B baseline run onwards
  (`eval/dataset/README.md`).

## Per-bug extra kinds

| Item | Bug | Extra kinds | Reason |
| --- | --- | --- | --- |
| item_001 | b1 | `missing_none_check` | The defect confuses "falsy" with "is None", so either label describes it |

## Rejected: a global `wrong_comparison` → `missing_none_check` group

Proposed in stage 0D for item_001 b1. Rejected because it would also loosen unrelated comparison
bugs: every planted `wrong_comparison` (`<` for `<=`, `==` for `!=`, a wrong constant) would then
accept a `missing_none_check` finding, which does not describe them.
