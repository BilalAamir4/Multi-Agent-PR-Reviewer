# Finding categories (D8)

The eight values of `Finding.category` (`src/reviewer/schemas.py`). The seven planted kinds are
defined as the dataset bugs were planted against them in stage 0D; `other` can never be a hit
(D8). Every agent prompt takes its category definitions from this file. A prompt phrases them as
what to look for in new or changed code, never as edits someone made, and keeps their meaning.

- `off_by_one`: a wrong boundary (< vs <=, range(n) vs range(n + 1), slice ends).
- `missing_none_check`: removing or omitting a None guard, so a reachable None causes an error or a wrong result.
- `wrong_comparison`: a wrong operator or operand (== vs is, and vs or, comparing the wrong variable, inverted condition).
- `unhandled_exception`: removing or narrowing a try/except (or a guard), so an expected error now propagates.
- `sql_injection`: user-controlled data formatted into SQL instead of passed as parameters, or quoting or escaping removed.
- `hardcoded_secret`: a credential literal where configuration or a parameter was used.
- `ineffective_test`: a test changed so it still passes but no longer checks the behavior (asserts the wrong thing, assertion removed or always true, wrong fixture).
- `other`: a real defect that fits none of the seven kinds above.
