"""Prices for the catalogue."""

PRICES = {
    "apple": 1.20,
    "pear": 1.50,
    "plum": 0.80,
    "fig": 2.40,
    "lime": 0.60,
    "mango": 3.10,
}

BULK_THRESHOLD = 10  # 10 or more units of one item earn the bulk discount
BULK_RATE = 0.05


def unit_price(sku):
    """Price of one unit, or None when the SKU is not in the catalogue."""
    return PRICES.get(sku)


def bulk_discount(qty):
    """Fraction taken off a line: 5% for 10 or more units, nothing below that."""
    if qty > BULK_THRESHOLD:
        return BULK_RATE
    return 0.0


def line_total(sku, qty):
    """Price of ``qty`` units of ``sku`` after the bulk discount; 0.0 for an unknown SKU."""
    price = unit_price(sku)
    if price is None:
        return 0.0
    gross = price * qty
    return round(gross * (1 - bulk_discount(qty)), 2)


def price_list(skus):
    """Map each SKU to its unit price; an unknown SKU maps to None."""
    return {sku: unit_price(sku) for sku in skus}


def cheapest(skus):
    """The known SKU with the lowest unit price, or None when none of them is known."""
    known = {sku: p for sku, p in price_list(skus).items() if p is not None}
    if not known:
        return None
    return min(known, key=known.get)


def basket_total(basket):
    """Total of a basket given as {sku: qty}, bulk discounts included."""
    return round(sum(line_total(sku, qty) for sku, qty in basket.items()), 2)


def describe(sku):
    """A short label such as ``apple (1.20)``; ``unknown`` for an unknown SKU."""
    price = unit_price(sku)
    if price is None:
        return "unknown"
    return f"{sku} ({price:.2f})"


COUPONS = {"WELCOME5": 5.0, "FRESH10": 10.0}


def apply_coupon(total, code):
    """Take the coupon's fixed amount off ``total``; an unknown code changes nothing."""
    amount = COUPONS.get(code, 0.0)
    return round(max(total - amount, 0.0), 2)


def price_range(skus):
    """(lowest, highest) unit price among the known SKUs, or None when none is known."""
    prices = [p for p in price_list(skus).values() if p is not None]
    if not prices:
        return None
    return (min(prices), max(prices))


def price_sheet(skus):
    """One text line per SKU, using ``describe``."""
    return "\n".join(describe(sku) for sku in skus)
