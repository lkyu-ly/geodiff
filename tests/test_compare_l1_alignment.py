"""Checks of the CPU comparator on small hand-built replay archives."""

from pathlib import Path

import numpy as np

from tools.compare_l1_alignment import NO_GRAD_PARAMETERS, compare, load, stats


EDGE_ENCODER = 'edge_encoder_global.bond_emb.weight'


def replay_arrays(no_grad):
    """The arrays a replay tool writes for one node, one edge and one gradient."""
    return dict(
        num_nodes=np.int64(1), loss=np.full((1, 1), 3.0), loss_global=np.full((1, 1), 2.0),
        loss_local=np.full((1, 1), 1.0), edge_index=np.array([[0], [0]]), edge_type=np.array([1]),
        local_edge_mask=np.array([True]), edge_length=np.full((1, 1), 1.5),
        **{f'grad::{EDGE_ENCODER}': np.full((2, 2), 0.25)},
        no_grad=np.array(no_grad), loss_mean=np.float64(3.0), grad_norm=np.float64(0.5),
    )


def write_replay(path, **changes):
    np.savez(path, **{**replay_arrays(list(NO_GRAD_PARAMETERS)), **changes})
    return Path(path)


def test_passes_when_both_sides_record_the_five_parameters_without_gradient(tmp_path):
    report, failures = compare(load(write_replay(tmp_path / 'torch.npz')),
                               load(write_replay(tmp_path / 'paddle.npz', grad_norm=np.float64(0.50002))))
    assert failures == []
    assert 'PASS' in '\n'.join(report)


def test_fails_when_the_torch_side_records_a_wrong_no_gradient_set(tmp_path):
    for wrong in ([], NO_GRAD_PARAMETERS[:4], [*NO_GRAD_PARAMETERS, EDGE_ENCODER]):
        report, failures = compare(load(write_replay(tmp_path / 'torch.npz', no_grad=np.array(wrong))),
                                   load(write_replay(tmp_path / 'paddle.npz')))
        assert any('torch no_grad' in failure for failure in failures), wrong
        assert 'FAIL' in '\n'.join(report)


def test_fails_when_the_paddle_side_records_a_wrong_no_gradient_set(tmp_path):
    report, failures = compare(load(write_replay(tmp_path / 'torch.npz')),
                               load(write_replay(tmp_path / 'paddle.npz',
                                                 no_grad=np.array(NO_GRAD_PARAMETERS[:4]))))
    assert any('paddle no_grad' in failure for failure in failures)
    assert 'FAIL' in '\n'.join(report)


def test_fails_when_a_recorded_scalar_is_outside_tolerance(tmp_path):
    for scalar in ('loss_mean', 'grad_norm'):
        report, failures = compare(load(write_replay(tmp_path / 'torch.npz')),
                                   load(write_replay(tmp_path / 'paddle.npz', **{scalar: np.float64(3.5)})))
        assert any(failure.startswith(scalar) for failure in failures), scalar
        assert 'FAIL' in '\n'.join(report)


def test_relative_error_is_reported_below_the_absolute_tolerance():
    # The pair passes on atol alone, but the entries have opposite signs.
    result = stats(np.array([5e-5]), np.array([-5e-5]))
    assert result['passed']
    assert result['max_rel_masked'] == 0
    assert result['max_rel'] == 2
