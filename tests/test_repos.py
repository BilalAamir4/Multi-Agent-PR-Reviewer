"""eval/repos.yaml loader, git output parsing and the pytest summary parser."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eval.check_repos import parse_pytest_summary
from eval.repos import REPOS_YAML, dataset_repos, load_repos, parse_ls_tree, sandbox_spec
from reviewer.tools import SandboxSpec

SHA = "0123456789abcdef0123456789abcdef01234567"


def spec(**overrides) -> dict:
    data = {
        "url": "https://github.com/owner/demo.git",
        "license": "MIT",
        "default_branch": "main",
        "image_commit": SHA,
        "image": f"mapr-sandbox-demo:{SHA[:12]}",
        "pythonpath": ["src"],
        "test_command": ["python", "-m", "pytest", "-q"],
        "test_timeout_s": 120,
    }
    data.update(overrides)
    return data


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "repos.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_real_file_loads():
    repos = load_repos(REPOS_YAML)
    assert set(repos) == {"sqlite-utils", "alembic", "click", "marshmallow", "dev_fixture"}
    assert dataset_repos() == ["sqlite-utils", "alembic", "click", "marshmallow"]
    assert repos["sqlite-utils"].slug == "simonw/sqlite-utils"
    for name in ("sqlite-utils", "alembic"):
        assert set(repos[name].recommended_plants) == {"sql_injection", "hardcoded_secret"}


def test_valid_minimal(tmp_path):
    repos = load_repos(write(tmp_path, {"repos": {"demo": spec()}}))
    assert repos["demo"].recommended_plants == [] and repos["demo"].eligible_since is None


def test_unknown_repo_key_rejected(tmp_path):
    with pytest.raises(ValidationError, match="extra"):
        load_repos(write(tmp_path, {"repos": {"demo": spec(network=True)}}))


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValidationError, match="extra"):
        load_repos(write(tmp_path, {"repos": {"demo": spec()}, "extra_key": 1}))


def test_image_tag_must_match_name_and_commit(tmp_path):
    with pytest.raises(ValidationError, match="image must be"):
        load_repos(write(tmp_path, {"repos": {"demo": spec(image="mapr-sandbox-demo:latest")}}))


@pytest.mark.parametrize(
    "override",
    [
        {"license": "GPL-3.0"},
        {"image_commit": "abc123"},
        {"url": "https://gitlab.com/owner/demo.git"},
        {"recommended_plants": ["other"]},
        {"recommended_plants": ["not_a_kind"]},
        {"test_command": []},
        {"test_timeout_s": 0},
    ],
)
def test_invalid_values_rejected(tmp_path, override):
    with pytest.raises(ValidationError):
        load_repos(write(tmp_path, {"repos": {"demo": spec(**override)}}))


def local(**overrides) -> dict:
    data = spec(
        url=None,
        license="n/a",
        default_branch=None,
        image_commit=None,
        image="mapr-sandbox-dev_fixture:local",
    )
    data.update(overrides)
    return data


def test_real_dev_fixture_entry():
    dev = load_repos(REPOS_YAML)["dev_fixture"]
    assert dev.is_local and dev.slug is None and dev.license == "n/a"
    assert sandbox_spec("dev_fixture") == SandboxSpec(
        image="mapr-sandbox-dev_fixture:local",
        test_command=("python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x"),
        pythonpath=(".",),
        timeout_s=60,
    )


def test_local_image_allowed_for_dev_fixture(tmp_path):
    repos = load_repos(write(tmp_path, {"repos": {"dev_fixture": local()}}))
    assert repos["dev_fixture"].url is None


@pytest.mark.parametrize(
    "override",
    [
        {"url": None},
        {"license": "n/a"},
        {"image_commit": None},
        {"default_branch": None},
    ],
)
def test_local_fields_only_for_local_names(tmp_path, override):
    with pytest.raises(ValidationError, match="only for the local images"):
        load_repos(write(tmp_path, {"repos": {"demo": spec(**override)}}))


@pytest.mark.parametrize(
    "override",
    [
        {"url": "https://github.com/owner/demo.git"},
        {"license": "MIT"},
        {"image_commit": SHA},
        {"image": "mapr-sandbox-dev_fixture:latest"},
    ],
)
def test_local_entry_must_be_fully_local(tmp_path, override):
    with pytest.raises(ValidationError):
        load_repos(write(tmp_path, {"repos": {"dev_fixture": local(**override)}}))


def test_parse_ls_tree():
    text = (
        b"100644 blob " + b"a" * 40 + b"     120\tsrc/pkg/a.py\0"
        b"120000 blob " + b"b" * 40 + b"       9\tlink\0"
        b"160000 commit " + b"c" * 40 + b"       -\tvendor/sub\0"
    )
    a, link, sub = parse_ls_tree(text)
    assert a == ("100644", "blob", "a" * 40, 120, "src/pkg/a.py")
    assert link[0] == "120000"
    assert sub[1] == "commit" and sub[3] is None


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (
            "....\n1497 passed, 16 skipped, 33 warnings in 8.37s\n",
            {"passed": 1497, "skipped": 16, "warnings": 33, "failed": 0},
        ),
        (
            "== 1 failed, 1754 passed, 23 skipped, 31000 deselected, 1 xfailed in 3.07s ==\n",
            {"failed": 1, "passed": 1754, "deselected": 31000, "xfailed": 1},
        ),
        ("=== 1 error in 0.56s ===\n", {"errors": 1, "passed": 0}),
        ("2 errors in 0.5s\n", {"errors": 2}),
    ],
)
def test_parse_pytest_summary(stdout, expected):
    counts = parse_pytest_summary("noise in 3.0s line without counts\n" + stdout)
    for key, value in expected.items():
        assert getattr(counts, key) == value
    assert counts.summary


def test_parse_pytest_summary_missing():
    counts = parse_pytest_summary("ImportError while loading conftest\n")
    assert counts.summary == "" and counts.passed == 0
