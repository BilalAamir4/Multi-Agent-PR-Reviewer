"""A shopping cart over a catalog."""

from shop.coupons import CouponBook


class Cart:
    def __init__(self, catalog, coupons=None):
        self.catalog = catalog
        self.coupons = coupons if coupons is not None else CouponBook()
        self.coupon = None
        self.lines = {}

    def add(self, sku, quantity=1):
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        if self.catalog.find(sku) is None:
            raise KeyError(sku)
        self.lines[sku] = self.lines.get(sku, 0) + quantity

    def apply_coupon(self, code):
        """Use the coupon with this code. Unknown codes are ignored; returns True if applied."""
        coupon = self.coupons.find(code)
        if coupon.percent > 0:
            self.coupon = coupon
            return True
        return False

    def subtotal_cents(self):
        total = 0
        for sku, quantity in self.lines.items():
            total += self.catalog.find(sku).price_cents * quantity
        return total

    def total_cents(self):
        total = self.subtotal_cents()
        if self.coupon is not None:
            total -= total * self.coupon.percent // 100
        return total
