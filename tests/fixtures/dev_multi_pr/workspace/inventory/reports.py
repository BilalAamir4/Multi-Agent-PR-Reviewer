"""Text reports over orders and stock."""

from inventory.pricing import unit_price
from inventory.stock import low_stock, restock_plan


def order_rows(order):
    """One row per known item of ``order`` ({sku: qty}): (sku, qty, unit price, line total).

    Items that are not in the catalogue are skipped."""
    rows = []
    for sku, qty in order.items():
        try:
            price = unit_price(sku)
        except KeyError:
            continue  # unknown items are skipped
        rows.append((sku, qty, price, round(price * qty, 2)))
    return rows


def format_rows(rows):
    lines = []
    for sku, qty, price, total in rows:
        lines.append(f"{sku:<8}{qty:>4} x {price:>6.2f} = {total:>8.2f}")
    return "\n".join(lines)


def order_report(order):
    """The order as text, with a total line."""
    rows = order_rows(order)
    total = round(sum(r[3] for r in rows), 2)
    body = format_rows(rows)
    return f"{body}\ntotal {total:.2f}" if body else "no known items"


def stock_report(levels, threshold):
    """One line per SKU below ``threshold``, with the units to reorder."""
    plan = restock_plan(levels, threshold)
    lines = [f"{sku}: order {plan[sku]}" for sku in low_stock(levels, threshold)]
    return "\n".join(lines) if lines else "stock is fine"


def top_items(order, n):
    """The ``n`` lines of ``order`` with the highest totals, highest first."""
    rows = sorted(order_rows(order), key=lambda r: r[3], reverse=True)
    return rows[:n]


def movement_report(history, n):
    """The last ``n`` movements as text, oldest first."""
    from inventory.stock import last_movements

    lines = [f"{m['sku']}: {m['delta']:+d}" for m in last_movements(history, n)]
    return "\n".join(lines) if lines else "no movements"


def price_sheet(skus):
    """A catalogue page for ``skus``: one row per known SKU, cheapest first."""
    rows = sorted(order_rows({sku: 1 for sku in skus}), key=lambda r: (r[2], r[0]))
    return format_rows(rows)
