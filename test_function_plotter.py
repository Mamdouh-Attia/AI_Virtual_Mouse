import plotter
import pytest

@pytest.mark.parametrize("character, sign , expected", [
    ('*', '*', True),
    ("/", "2", False),
    ("5", 1, False),
])
def test_valid_sign(character, sign,  expected):
    assert plotter.Ui_MainWindow().isNotValid(character,sign) == expected

@pytest.mark.parametrize("num , expected", [
    ('5', True),
    ('x', False),
    ("%", False),
])
def test_is_valid_function(num , expected):
    assert plotter.Ui_MainWindow().check_if_number(num) == expected

@pytest.mark.parametrize("function , expected", [
    ('x+1', False),
    ('2/x*5', False),
    ("1+2-1+20", True),
])
def test_is_valid_function(function , expected):
    assert plotter.Ui_MainWindow().isAllNumeric(function) == expected
