# Categories

Give each finding exactly one category:

- `off_by_one`: a boundary is wrong (< vs <=, range(n) vs range(n + 1), slice ends).
- `missing_none_check`: a None guard is missing or was removed, so a reachable None causes an
  error or a wrong result.
- `wrong_comparison`: an operator or operand is wrong (== vs is, and vs or, comparing the wrong
  variable, an inverted condition).
- `unhandled_exception`: a try/except (or a guard) is missing, was removed or is too narrow, so
  an expected error propagates.
- `sql_injection`: user-controlled data is formatted into SQL instead of passed as parameters,
  or quoting or escaping is missing.
- `hardcoded_secret`: a credential literal is in the code where configuration or a parameter
  should be used.
- `ineffective_test`: a test still passes but no longer checks the behaviour (it asserts the
  wrong thing, an assertion is missing or always true, or it uses the wrong fixture).
- `other`: a real defect that fits none of the seven kinds above.
