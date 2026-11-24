"""Validation and application of the human's decisions (decision 0016).

A decision is ``{"finding_id": str, "action": "approve"|"edit"|"reject", "edited": {...}}``.
Every finding id of the interrupt payload appears exactly once. An edit may change only
``title``, ``explanation`` and ``severity``; never the file, the lines or the category."""

from __future__ import annotations

from typing import Any

ACTIONS = ("approve", "edit", "reject")
EDITABLE = ("title", "explanation", "severity")
SEVERITIES = ("high", "medium", "low")


class DecisionError(ValueError):
    """The decisions do not match the payload or break a rule."""


def validate_decisions(decisions: Any, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the decisions normalized and ordered like the payload, or raise ``DecisionError``."""
    if not isinstance(decisions, list):
        raise DecisionError("decisions must be a list")
    ids = [e["id"] for e in payload]
    seen: dict[str, dict[str, Any]] = {}
    for d in decisions:
        if not isinstance(d, dict):
            raise DecisionError("every decision must be an object")
        fid = d.get("finding_id")
        if fid not in ids:
            raise DecisionError(f"unknown finding id: {fid!r}")
        if fid in seen:
            raise DecisionError(f"duplicate decision for finding {fid}")
        action = d.get("action")
        if action not in ACTIONS:
            raise DecisionError(f"finding {fid}: action must be one of {', '.join(ACTIONS)}")
        edited = d.get("edited") or {}
        if not isinstance(edited, dict):
            raise DecisionError(f"finding {fid}: 'edited' must be an object")
        bad = sorted(set(edited) - set(EDITABLE))
        if bad:
            raise DecisionError(
                f"finding {fid}: cannot edit {', '.join(bad)} (only {', '.join(EDITABLE)})"
            )
        if action != "edit" and edited:
            raise DecisionError(f"finding {fid}: 'edited' is only for action 'edit'")
        if action == "edit":
            if not edited:
                raise DecisionError(f"finding {fid}: an edit must change something")
            for key in ("title", "explanation"):
                if key in edited and (not isinstance(edited[key], str) or not edited[key].strip()):
                    raise DecisionError(f"finding {fid}: {key} must not be blank")
            if "severity" in edited and edited["severity"] not in SEVERITIES:
                raise DecisionError(
                    f"finding {fid}: severity must be one of {', '.join(SEVERITIES)}"
                )
        seen[fid] = {"finding_id": fid, "action": action, "edited": dict(edited)}
    missing = [i for i in ids if i not in seen]
    if missing:
        raise DecisionError(f"missing decision for finding(s): {', '.join(missing)}")
    return [seen[i] for i in ids]


def apply_decisions(
    decisions: list[dict[str, Any]], payload: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The approved findings (payload entries, edits applied, ``edited`` flag), in payload order.
    ``decisions`` must already be valid."""
    by_id = {d["finding_id"]: d for d in decisions}
    approved = []
    for entry in payload:
        d = by_id[entry["id"]]
        if d["action"] == "reject":
            continue
        item = dict(entry)
        item["edited"] = d["action"] == "edit"
        if d["action"] == "edit":
            item.update(d["edited"])
        approved.append(item)
    return approved
