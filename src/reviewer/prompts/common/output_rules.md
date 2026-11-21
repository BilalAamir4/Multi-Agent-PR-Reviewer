# Output rules

- Only report defects in lines this pull request adds or changes.
- Cite new-file line numbers exactly as `read_diff` shows them.
- `line_start` and `line_end` cover only the defective lines.
- `evidence` is one to five lines of code copied exactly from the file. Copy only the code:
  leave out the `NN | ` prefix that `read_file` adds, and the line number and the `+` or ` `
  marker that `read_diff` adds. For example, when `read_file` shows `  12 |     return total`,
  the evidence is `    return total`, not `12 |     return total`.
- Report one finding per distinct defect.
- An empty findings list is a correct answer when the change has no defects, so do not invent
  findings.
- `summary` is 2 to 4 sentences on what the change does.
