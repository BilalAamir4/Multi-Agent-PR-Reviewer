from geometry.shapes import average_side, rectangle_area, rectangle_perimeter, triangle_area


def test_rectangle_area():
    assert rectangle_area(3, 4) == 12


def test_triangle_area():
    assert triangle_area(3, 4) == 6


def test_rectangle_perimeter():
    assert rectangle_perimeter(3, 4) > 0


def test_average_side():
    assert average_side(3, 4) == 3.5
