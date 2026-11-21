from inventory.reports import format_rows, order_report, order_rows, stock_report, top_items
from inventory.stock import (
    apply_movement,
    last_movements,
    level,
    low_stock,
    net_change,
    record,
    replay,
    restock_plan,
)


def test_order_rows_for_known_items():
    rows = order_rows({"apple": 2, "pear": 1})
    assert rows == [("apple", 2, 1.20, 2.4), ("pear", 1, 1.50, 1.5)]


def test_format_rows():
    text = format_rows([("apple", 2, 1.20, 2.4)])
    assert text == "apple      2 x   1.20 =     2.40"


def test_order_report_has_a_total():
    assert order_report({"apple": 2, "pear": 2}).endswith("total 5.40")


def test_top_items_orders_by_total():
    assert [r[0] for r in top_items({"apple": 1, "mango": 2, "plum": 1}, 2)] == ["mango", "apple"]


def test_stock_flow():
    levels = {}
    history = []
    for sku, delta in [("apple", 10), ("pear", 4), ("apple", -7)]:
        apply_movement(levels, sku, delta)
        record(history, sku, delta)
    assert level(levels, "apple") == 3
    assert low_stock(levels, 5) == ["apple", "pear"]
    assert restock_plan(levels, 5) == {"apple": 2, "pear": 1}
    assert net_change(history, "apple") == 3
    assert replay(history) == levels
    assert stock_report(levels, 5) == "apple: order 2\npear: order 1"


def test_no_movements_to_show():
    assert last_movements([], 0) == []


def test_price_sheet_is_cheapest_first():
    from inventory.reports import price_sheet

    first = price_sheet(["fig", "lime"]).splitlines()[0]
    assert first.startswith("lime")


def test_most_moved_and_merge():
    from inventory.stock import merge_levels, most_moved, sold_out

    history = [
        {"sku": "apple", "delta": 5},
        {"sku": "pear", "delta": -2},
        {"sku": "apple", "delta": -1},
    ]
    assert most_moved(history, 1) == ["apple"]
    assert merge_levels({"apple": 1}, {"apple": 2, "pear": 3}) == {"apple": 3, "pear": 3}
    assert sold_out({"apple": 0, "pear": 4}) == ["apple"]
