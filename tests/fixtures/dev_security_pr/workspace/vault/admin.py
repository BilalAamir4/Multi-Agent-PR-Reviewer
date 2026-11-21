"""Login for the admin console."""

import hmac

ADMIN_USER = "admin"
ADMIN_PASSWORD = "Winter-Harbor-2026"


def check_login(user, password):
    """True if ``user`` and ``password`` are the admin's."""
    return user == ADMIN_USER and hmac.compare_digest(password, ADMIN_PASSWORD)
