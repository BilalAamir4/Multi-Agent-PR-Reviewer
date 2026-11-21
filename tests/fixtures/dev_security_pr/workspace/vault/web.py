"""Request handlers for the user directory."""

from vault import admin, db


def handle_lookup(conn, params):
    user = db.find_user(conn, params["name"])
    return {"found": user is not None}


def handle_search(conn, params):
    rows = db.search_users(conn, params.get("q", ""), params.get("sort", "name"))
    return {"users": [{"id": r[0], "name": r[1]} for r in rows]}


def handle_admin_login(params):
    return {"ok": admin.check_login(params.get("user", ""), params.get("password", ""))}
