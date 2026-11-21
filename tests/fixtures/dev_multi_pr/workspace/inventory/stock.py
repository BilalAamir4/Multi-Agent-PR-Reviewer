"""Stock levels and the movements that change them."""


def apply_movement(levels, sku, delta):
    """Add ``delta`` (negative for a sale) to the level of ``sku``."""
    levels[sku] = levels.get(sku, 0) + delta
    return levels


def level(levels, sku):
    """Units of ``sku`` in stock; zero for an item never seen."""
    return levels.get(sku, 0)


def record(history, sku, delta):
    """Append one movement to ``history`` and return it."""
    history.append({"sku": sku, "delta": delta})
    return history


def last_movements(history, n):
    """The ``n`` most recent movements, oldest first."""
    if n <= 0:
        return []
    return history[len(history) - n + 1 :]


def low_stock(levels, threshold):
    """SKUs whose level is below ``threshold``, in alphabetical order."""
    return sorted(sku for sku, units in levels.items() if units < threshold)


def restock_plan(levels, target):
    """Units to order per SKU so every level reaches ``target``; only SKUs below it appear."""
    plan = {}
    for sku, units in levels.items():
        if units < target:
            plan[sku] = target - units
    return plan


def net_change(history, sku):
    """Sum of the movements of one SKU."""
    return sum(m["delta"] for m in history if m["sku"] == sku)


def replay(history):
    """Rebuild the stock levels from a movement history."""
    levels = {}
    for m in history:
        apply_movement(levels, m["sku"], m["delta"])
    return levels


def most_moved(history, n=3):
    """The ``n`` SKUs with the most units moved in either direction, busiest first."""
    moved = {}
    for m in history:
        moved[m["sku"]] = moved.get(m["sku"], 0) + abs(m["delta"])
    ranked = sorted(moved, key=lambda sku: (-moved[sku], sku))
    return ranked[:n]


def merge_levels(first, second):
    """Combine two stock tables by adding the levels of SKUs present in both."""
    merged = dict(first)
    for sku, units in second.items():
        merged[sku] = merged.get(sku, 0) + units
    return merged


def sold_out(levels):
    """SKUs whose level is zero or below, in alphabetical order."""
    return sorted(sku for sku, units in levels.items() if units <= 0)


def movements_for(conn, sku):
    """Recorded movements of ``sku`` from the stock database, oldest first."""
    query = f"SELECT sku, delta FROM movements WHERE sku = '{sku}' ORDER BY id"
    return conn.execute(query).fetchall()
