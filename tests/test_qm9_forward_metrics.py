import numpy as np
import pytest

from tools import check_paddle_forward


def test_forward_metrics_use_both_magnitudes_for_sign_flips():
    actual = np.array([-2.0, 3.0, 0.0], dtype=np.float32)
    expected = np.array([1.0, -1.0, 0.0], dtype=np.float32)

    assert check_paddle_forward.error_metrics(actual, expected) == pytest.approx(
        {'max_abs': 4.0, 'mean_abs': 7 / 3, 'mean_rel': 17 / 18}, rel=1e-14, abs=0
    )


def test_forward_metrics_floor_the_denominator_near_zero():
    actual = np.array([0.0, 2.5e-31])
    expected = np.zeros(2)

    assert check_paddle_forward.error_metrics(actual, expected) == pytest.approx(
        {'max_abs': 2.5e-31, 'mean_abs': 1.25e-31, 'mean_rel': 0.125}, rel=1e-14, abs=0
    )


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
def test_forward_metrics_reject_matching_nonfinite_outputs(nonfinite):
    actual = np.array([0.5, nonfinite], dtype=np.float32)
    expected = actual.copy()

    with pytest.raises(ValueError, match='finite'):
        check_paddle_forward.error_metrics(actual, expected)
