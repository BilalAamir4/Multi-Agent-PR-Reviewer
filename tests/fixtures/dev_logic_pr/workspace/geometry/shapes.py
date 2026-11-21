"""Areas and perimeters of simple shapes."""


def rectangle_area(width: float, height: float) -> float:
    return width * height


def triangle_area(base: float, height: float) -> float:
    return base * height / 2


def rectangle_perimeter(width: float, height: float) -> float:
    return width + height


def average_side(width: float, height: float) -> float:
    return rectangle_perimeter(width, height) / 4
