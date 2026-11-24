"""Exercise the parity commands with small CPU fixtures."""

import importlib.util
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


def tiny_tape():
    return dict(
        num_graphs=np.array(1, dtype=np.int64),
        num_nodes_per_graph=np.array([3], dtype=np.int64),
        batch=np.array([0, 0, 0], dtype=np.int64),
        atom_type=np.array([6, 8, 6], dtype=np.int64),
        pos=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        bond_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        bond_type=np.array([1, 1], dtype=np.int64),
        time_step=np.full((20, 1), 100, dtype=np.int64),
        pos_noise=np.zeros((20, 3, 3), dtype=np.float32),
    )


def compare(tmp_path, torch_steps, paddle_steps, *, swapped=False, different_field=None):
    torch, paddle, output = (tmp_path / f'{name}.json' for name in ('torch', 'paddle', 'out'))
    for path, backend, steps in ((torch, 'torch', torch_steps), (paddle, 'paddle', paddle_steps)):
        arrays = tiny_tape()
        if backend == 'paddle' and different_field is not None:
            if different_field == 'pos_noise':
                arrays[different_field][-1, 0, 0] = -0.0
            else:
                arrays[different_field].flat[0] += 1
        sidecar = path.with_name(path.name + '.inputs.npz')
        np.savez(sidecar, **arrays)
        path.write_text(json.dumps(dict(backend=backend, inputs=sidecar.name, steps=steps)))
    if swapped:
        torch, paddle = paddle, torch
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


def test_compare_refuses_swapped_backend_files(tmp_path):
    result, out = compare(tmp_path, trajectory(), trajectory(), swapped=True)

    assert result.returncode == 2
    assert 'backend' in result.stderr and 'torch' in result.stderr
    assert out is None


@pytest.mark.parametrize('field', ['pos', 'time_step', 'pos_noise'])
def test_compare_refuses_same_shape_tapes_with_different_bytes(tmp_path, field):
    result, out = compare(tmp_path, trajectory(), trajectory(), different_field=field)

    assert result.returncode == 2
    assert 'input tape' in result.stderr and field in result.stderr
    assert out is None


@pytest.mark.skipif(importlib.util.find_spec('paddle') is None, reason='requires native Paddle')
def test_paddle_cpu_runner_updates_both_optimizer_groups(tmp_path):
    arrays = tiny_tape()
    tape, weights, output = (tmp_path / name for name in ('tape.npz', 'tiny.pdparams', 'rows.json'))
    np.savez(tape, **arrays, unused=np.arange(64))
    # With only the two final biases nonzero, each branch depends on its own
    # group alone. The three atoms give one bond and two radius-only pairs.
    root = TOOL.parents[1] / 'geodiff_paddle'
    make_weights = """
import sys
from pathlib import Path
import paddle
import yaml
from easydict import EasyDict

sys.path.insert(0, sys.argv[1])
from models.epsnet import get_model

paddle.set_device('cpu')
config = EasyDict(yaml.safe_load((Path(sys.argv[1]) / 'configs/qm9_default.yml').read_text()))
model = get_model(config.model)
with paddle.no_grad():
    for parameter in model.parameters():
        parameter.set_value(paddle.zeros_like(parameter))
    model.grad_global_dist_mlp.layers[-1].bias.set_value(paddle.full([1], 0.75))
    model.grad_local_dist_mlp.layers[-1].bias.set_value(paddle.full([1], 0.375))
paddle.save(model.state_dict(), sys.argv[2])
"""
    fixture = subprocess.run([sys.executable, '-c', make_weights, str(root), str(weights)],
                             capture_output=True, text=True, check=False)
    assert fixture.returncode == 0, fixture.stderr

    result = run('paddle', '--device', 'cpu', '--inputs', tape, '--weights', weights, '--out', output)
    assert result.returncode == 0, result.stderr
    archive = json.loads(output.read_text())
    assert archive['backend'] == 'paddle'
    with np.load(output.with_name(archive['inputs']), allow_pickle=False) as snapshot:
        assert set(snapshot.files) == arrays.keys()
        for name, original in arrays.items():
            assert snapshot[name].dtype == original.dtype and snapshot[name].shape == original.shape, name
            assert snapshot[name].tobytes() == original.tobytes(), name
    rows = archive['steps']
    assert [row['step'] for row in rows] == list(range(1, 21))
    assert rows[0]['loss_global'] == pytest.approx((32 + 8 * np.sqrt(2)) / 3 * 0.75 ** 2)
    assert rows[0]['loss_local'] == pytest.approx(40 / 3 * 0.375 ** 2)
    # Three pre-update rows with identical input reveal the first two updates.
    for field in ('loss_global', 'loss_local'):
        assert rows[0][field] > rows[1][field] > rows[2][field], field


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
