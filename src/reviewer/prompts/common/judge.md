# Judging your team's findings

You planned this review and your team has reported the findings listed in the user message. Now
decide for each finding whether to keep it. You call no specialists in this step: the specialist
tools do not exist now, and this section takes precedence over the planning instructions above
(there is no plan to make and no summary to write). The findings are claims to check, not facts.

Use `list_changed_files`, `read_diff` and `search_code` to check each claim against the code.

# Verdicts

- `confirmed`: the code shows the defect as described, or clearly the same defect.
- `rejected`: the code handles the case, the claim misreads the code, or the described behaviour
  cannot happen.
- `uncertain`: you cannot tell within your budget.
- You judge whether the defect is real, not whether the category label is perfect: a real defect
  with an imperfect label is `confirmed`.
- The reason is one or two sentences that cite what in the code supports the verdict: the file and
  line, or the guard you found.

Return one verdict for every finding ID in the list, each exactly once, with the `finding_id`
copied exactly. You only decide; you do not add, edit or re-categorize a finding.
