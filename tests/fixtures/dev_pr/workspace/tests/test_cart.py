import pytest

from shop.cart import Cart
from shop.catalog import Catalog, Product
from shop.coupons import Coupon, CouponBook


def make_catalog():
    return Catalog([Product("a", "Apple", 50), Product("b", "Bread", 200)])


def test_total():
    cart = Cart(make_catalog())
    cart.add("a", 3)
    cart.add("b")
    assert cart.total_cents() == 350


def test_unknown_sku():
    cart = Cart(make_catalog())
    with pytest.raises(KeyError):
        cart.add("zzz")


def test_coupon_discount():
    coupons = CouponBook([Coupon("SAVE10", 10)])
    cart = Cart(make_catalog(), coupons)
    cart.add("b", 2)
    assert cart.apply_coupon("save10") is True
    assert cart.total_cents() == 360


def test_page_count():
    assert make_catalog().page_count(1) == 2
