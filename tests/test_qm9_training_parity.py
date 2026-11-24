"""Run the CPU commands against small flat-batch and trajectory fixtures."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


TOOL = Path(__file__).resolve().parents[1] / 'tools/run_qm9_training_parity.py'


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_prepare_preserves_recorded_batch_and_replays_ordered_seeded_draws(tmp_path):
    original = dict(
        num_graphs=np.array(4, dtype=np.int64),
        num_nodes_per_graph=np.array([1, 2, 1, 1], dtype=np.int64),
        batch=np.array([0, 1, 1, 2, 3], dtype=np.int64),
        atom_type=np.array([6, 8, 6, 7, 9], dtype=np.int64),
        pos=np.arange(15, dtype=np.float32).reshape(5, 3),
        bond_index=np.array([[1, 2], [2, 1]], dtype=np.int64),
        bond_type=np.array([1, 1], dtype=np.int64),
        time_step=np.array([8, 9, 10, 4991], dtype=np.int64),
        pos_noise=np.full((5, 3), -0.0, dtype=np.float32),
    )
    recorded, tape = tmp_path / 'recorded.npz', tmp_path / 'tape.npz'
    np.savez(recorded, **original)

    result = run('prepare', '--inputs', recorded, '--out', tape)
    assert result.returncode == 0, result.stderr
    with np.load(tape, allow_pickle=False) as output:
        for name, values in original.items():
            data = output[name][0] if name in ('time_step', 'pos_noise') else output[name]
            assert data.shape == values.shape and data.dtype == values.dtype, name
            assert data.tobytes() == values.tobytes(), name
        t, noise = output['time_step'], output['pos_noise']
        assert t.shape == (20, 4)
        assert noise.shape == (20, 5, 3)
        assert ((t >= 0) & (t < 5000)).all()
        np.testing.assert_array_equal(t[1:, 0] + t[1:, 3], 4999)
        np.testing.assert_array_equal(t[[1, 2, 19]], [[3788, 3784, 2491, 1211],
                                                      [4735, 3609, 1591, 264],
                                                      [4404, 667, 2695, 595]])
        np.testing.assert_array_equal(noise[[1, 2, 19], 0].view(np.uint32),
                                      [[1075418668, 3207921180, 1060545274],
                                       [1068102365, 3198329716, 1066716703],
                                       [3211455117, 1065024073, 3195604081]])

    before = recorded.read_bytes()
    result = run('prepare', '--inputs', recorded, '--out', recorded)
    assert result.returncode != 0
    assert recorded.read_bytes() == before


def trajectory():
    return [dict(step=i, loss=8.0, loss_global=2.0, loss_local=6.0,
                 grad_norm=124.5, lr_global=0.001, lr_local=0.001) for i in range(1, 21)]


def compare(tmp_path, torch_steps, paddle_steps):
    torch, paddle, output = (tmp_path / f'{name}.json' for name in ('torch', 'paddle', 'out'))
    torch.write_text(json.dumps(torch_steps))
    paddle.write_text(json.dumps(paddle_steps))
    result = run('compare', '--torch', torch, '--paddle', paddle, '--out', output)
    return result, json.loads(output.read_text()) if output.exists() else None


def test_compare_reports_signed_errors_and_accepts_total_cross_zero(tmp_path):
    torch, paddle = trajectory(), trajectory()
    for row in paddle:
        row.update(loss_global=3.0, loss_local=5.0, grad_norm=123.5)
    paddle[0].update(loss=10.0, loss_local=7.0)
    paddle[1].update(loss=4.0, loss_local=1.0)
    result, out = compare(tmp_path, torch, paddle)

    assert result.returncode == 0, result.stderr
    assert out['passed'] is True
    assert [row['step'] for row in out['steps']] == list(range(1, 21))
    first = out['steps'][0]
    assert first['torch']['grad_norm'] == 124.5 and first['paddle']['grad_norm'] == 123.5
    assert first['torch']['lr_global'] == first['paddle']['lr_local'] == 0.001
    assert first['delta'] == dict(loss=2.0, loss_global=1.0, loss_local=1.0)
    assert out['steps'][1]['abs_error']['loss'] == 4.0
    assert out['steps'][1]['rel_error']['loss'] == 0.5
    total = out['metrics']['loss']
    assert total == pytest.approx(dict(mean_abs=0.3, mean_rel=0.035, min_diff=-4.0,
                                       max_diff=2.0, positive=1, negative=1, zero=18))
    assert out['metrics']['loss_global'] == pytest.approx(
        dict(mean_abs=1.0, mean_rel=1 / 3, min_diff=1.0, max_diff=1.0,
             positive=20, negative=0, zero=0))
    assert out['metrics']['loss_local'] == pytest.approx(
        dict(mean_abs=1.2, mean_rel=(1 / 7 + 5 / 6 + 18 / 6) / 20, min_diff=-5.0, max_diff=1.0,
             positive=1, negative=19, zero=0))


def test_compare_keeps_failed_total_verdict_even_when_branches_cross_zero(tmp_path):
    torch, paddle = trajectory(), trajectory()
    paddle[0].update(loss=10.0, loss_global=1.0, loss_local=9.0)
    paddle[1].update(loss_global=3.0, loss_local=5.0)
    result, out = compare(tmp_path, torch, paddle)

    assert result.returncode == 1, result.stderr
    assert out['passed'] is False
    assert out['metrics']['loss']['min_diff'] == 0.0
    assert out['metrics']['loss']['max_diff'] == 2.0


def test_compare_uses_float64_and_applies_the_relative_error_floor(tmp_path):
    torch, paddle = trajectory(), trajectory()
    paddle[0]['loss'] = 8.000000000000002
    torch[1]['loss'], paddle[1]['loss'] = 2e-32, 1e-32
    result, out = compare(tmp_path, torch, paddle)

    assert result.returncode == 0, result.stderr
    assert out['steps'][0]['delta']['loss'] == 2 ** -49
    assert out['steps'][1]['rel_error']['loss'] == pytest.approx(0.01)


@pytest.mark.parametrize('bad_rows, field', [
    ('short', '20'), ('duplicate', 'step'), ('reordered', 'step'), ('fractional', 'step'),
    ('nan', 'loss'), ('infinity', 'grad_norm'), ('missing', 'lr_local'), ('bool', 'loss_global'),
])
def test_compare_refuses_invalid_step_records(tmp_path, bad_rows, field):
    torch, paddle = trajectory(), trajectory()
    if bad_rows == 'short':
        paddle.pop()
    elif bad_rows == 'duplicate':
        paddle[1]['step'] = 1
    elif bad_rows == 'reordered':
        paddle[0], paddle[1] = paddle[1], paddle[0]
    elif bad_rows == 'fractional':
        paddle[1]['step'] = 2.0
    elif bad_rows == 'nan':
        paddle[1]['loss'] = float('nan')
    elif bad_rows == 'infinity':
        paddle[1]['grad_norm'] = float('inf')
    elif bad_rows == 'missing':
        del paddle[1]['lr_local']
    else:
        paddle[1]['loss_global'] = True

    result, out = compare(tmp_path, torch, paddle)
    assert result.returncode != 0
    assert 'paddle' in result.stderr and field in result.stderr
    assert out is None
