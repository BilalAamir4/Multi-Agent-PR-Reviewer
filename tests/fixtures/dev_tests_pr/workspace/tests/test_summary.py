import pytest

from stats.summary import clamp, mean, median


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_empty():
    try:
        mean([])
    except ValueError:
        pass


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


def test_clamp_bad_range():
    with pytest.raises(ValueError):
        clamp(1, 10, 0)
