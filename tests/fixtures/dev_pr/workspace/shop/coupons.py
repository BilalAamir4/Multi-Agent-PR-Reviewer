"""Percentage-off coupons."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Coupon:
    code: str
    percent: int


class CouponBook:
    def __init__(self, coupons=()):
        self._coupons = {}
        for coupon in coupons:
            if not 0 < coupon.percent <= 100:
                raise ValueError(f"bad percent for {coupon.code!r}")
            self._coupons[coupon.code.upper()] = coupon

    def find(self, code):
        """The coupon for this code (case-insensitive), or None if there is none."""
        return self._coupons.get(code.strip().upper())
