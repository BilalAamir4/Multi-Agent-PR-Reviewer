from inventory.pricing import (
    basket_total,
    bulk_discount,
    cheapest,
    describe,
    line_total,
    price_list,
    unit_price,
)


def test_unit_price_of_a_known_item():
    assert unit_price("apple") == 1.20


def test_unknown_item_has_no_price():
    assert unit_price("durian") is None


def test_line_total():
    assert line_total("pear", 4) == 6.0


def test_unknown_line_is_free():
    assert line_total("durian", 3) == 0.0


def test_bulk_discount_applies_to_a_big_order():
    assert bulk_discount(20) == 0.05
    assert line_total("apple", 20) == 22.8


def test_no_bulk_discount_for_a_small_order():
    assert bulk_discount(3) == 0.0


def test_price_list_marks_unknown_items():
    assert price_list(["fig", "durian"]) == {"fig": 2.40, "durian": None}


def test_cheapest_known_item():
    assert cheapest(["fig", "lime", "durian"]) == "lime"
    assert cheapest(["durian"]) is None


def test_basket_total():
    assert basket_total({"apple": 2, "plum": 5}) == 6.4


def test_describe():
    assert describe("fig") == "fig (2.40)"
    assert describe("durian") == "unknown"


def test_coupons():
    from inventory.pricing import apply_coupon

    assert apply_coupon(20.0, "FRESH10") == 10.0
    assert apply_coupon(3.0, "WELCOME5") == 0.0
    assert apply_coupon(20.0, "NOPE") == 20.0


def test_price_range():
    from inventory.pricing import price_range

    assert price_range(["fig", "lime", "durian"]) == (0.60, 2.40)
    assert price_range(["durian"]) is None
