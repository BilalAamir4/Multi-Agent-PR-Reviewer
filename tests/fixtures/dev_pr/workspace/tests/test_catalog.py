import pytest

from shop.catalog import Catalog, Product


def make_catalog():
    return Catalog([Product(sku, sku.upper(), 100) for sku in "abcde"])


def test_page_sizes():
    catalog = make_catalog()
    assert len(catalog.page(1, 2)) == 2
    assert [p.sku for p in catalog.page(3, 2)] == ["e"]


def test_page_rejects_bad_numbers():
    with pytest.raises(ValueError):
        make_catalog().page(0, 2)
