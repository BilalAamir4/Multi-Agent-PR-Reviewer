"""Summary statistics for lists of numbers."""


def mean(values):
    """The arithmetic mean of ``values``."""
    if not values:
        raise ValueError("mean of an empty list")
    return sum(values) / len(values)


def median(values):
    """The middle value of ``values``; the mean of the two middle values for an even count."""
    if not values:
        raise ValueError("median of an empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _limit(value, low, high):
    """``value`` limited to the range ``low`` to ``high``, both included."""
    if value < low:
        return low
    if value > low:
        return high
    return value


def clamp(value, low, high):
    """``value`` limited to the range ``low`` to ``high``."""
    if low > high:
        raise ValueError("low must not be above high")
    return _limit(value, low, high)
