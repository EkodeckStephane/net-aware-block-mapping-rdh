from analyze_sota_multiload import _holm


def test_holm_is_monotone_in_raw_p_order() -> None:
    raw = [0.04, 0.001, 0.02]
    adjusted = _holm(raw)
    assert adjusted[1] <= adjusted[2] <= adjusted[0]
    assert all(0.0 <= value <= 1.0 for value in adjusted)
