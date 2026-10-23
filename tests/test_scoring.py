import math

from score import parse, wilson


def test_parse_variants():
    assert parse('```json\n{"earlier": "A", "elapsed_hours": 2}\n```') == ("A", 2.0)
    assert parse('thinking...\n{"earlier": "B", "elapsed_hours": 1.5}') == ("B", 1.5)
    assert parse("no json here") == (None, None)
    assert parse('{"earlier": "A", "elapsed_hours": 1} then {"earlier": "B", "elapsed_hours": 4}') == ("B", 4.0)  # last wins


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert math.isclose(lo, 0.4038, abs_tol=1e-3) and math.isclose(hi, 0.5962, abs_tol=1e-3)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi < 0.35
