from vault import db


def make_db():
    conn = db.connect(":memory:")
    db.create_tables(conn)
    db.add_user(conn, "ada", "ada@example.org")
    db.add_user(conn, "grace", "grace@example.org")
    return conn


def test_find_user():
    assert db.find_user(make_db(), "ada")[2] == "ada@example.org"


def test_find_missing_user():
    assert db.find_user(make_db(), "linus") is None


def test_search_users():
    names = [row[1] for row in db.search_users(make_db(), "a")]
    assert names == ["ada", "grace"]
