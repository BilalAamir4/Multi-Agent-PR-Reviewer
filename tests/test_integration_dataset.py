"""Integration: every check in eval/check_dataset.py passes for every dataset item (Docker)."""

import pytest

from eval.build_dataset import all_item_ids, load_recipe
from eval.check_dataset import check_item
from eval.repos import clone_dir, get_repo
from reviewer.sandbox import SandboxUnavailable, get_client

pytestmark = pytest.mark.integration


def test_check_dataset_all_items():
    items = all_item_ids()
    assert len(items) == 5, items
    try:
        client = get_client()
    except SandboxUnavailable as e:
        pytest.skip(str(e))
    try:
        for item_id in items:
            name = load_recipe(item_id).repo
            spec = get_repo(name)
            try:
                client.images.get(spec.image)
            except Exception:
                pytest.skip(f"image {spec.image} not built: see sandbox_images/{name}")
            if not (clone_dir(name) / ".git").exists():
                pytest.skip(f"no clone at {clone_dir(name)}: run eval.repos.ensure_clone({name!r})")
    finally:
        client.close()

    failures = []
    for item_id in items:
        results = check_item(item_id)
        assert [r.number for r in results] == list(range(1, 13)), [r.line(item_id) for r in results]
        failures += [r.line(item_id) for r in results if r.ok is not True]
    assert failures == []
