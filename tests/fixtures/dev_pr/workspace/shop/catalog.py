"""Products and the catalog that holds them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int


class Catalog:
    def __init__(self, products=()):
        self._products = {p.sku: p for p in products}

    def add(self, product):
        if product.sku in self._products:
            raise ValueError(f"duplicate sku {product.sku!r}")
        self._products[product.sku] = product

    def find(self, sku):
        """The product with this sku, or None."""
        return self._products.get(sku)

    def all(self):
        """Every product, sorted by sku."""
        return [self._products[k] for k in sorted(self._products)]

    def page(self, number, size):
        """Products on page ``number`` (1-based) with ``size`` products per page, by sku."""
        if number < 1 or size < 1:
            raise ValueError("number and size must be at least 1")
        products = self.all()
        start = (number - 1) * size
        return products[start : start + size - 1]

    def page_count(self, size):
        """How many pages of ``size`` products the catalog fills."""
        return -(-len(self._products) // size)
