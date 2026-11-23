"""Opt-in CUDA smoke on the exported official two-example QM9 splits."""

import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def run_trainer(config, logdir, *options):
    logdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(ROOT / 'geodiff_paddle/train.py'), str(config),
         '--logdir', str(logdir), '--val_batches', '1', '--record_inputs', *options],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run, = logdir.iterdir()
    return run


def read_metrics(run, iteration):
    train, val = [line.split('] ', 1)[1] for line in (run / 'log.txt').read_text().splitlines()
                  if '[Train]' in line or '[Validate]' in line]
    print(train)
    print(val)
    assert f'[Train] Iter {iteration:05d}' in train
    assert f'[Validate] Iter {iteration:05d}' in val
    train_values = [float(value) for value in re.findall(r'\| [^ ]+ ([^ ]+)', train)]
    val_values = [float(value) for value in re.findall(r'\| [^ ]+ ([^ ]+)', val)]
    assert len(train_values) == 6 and len(val_values) == 3
    assert all(map(math.isfinite, train_values + val_values))
    assert all(value > 0 for value in train_values[1:])
    with np.load(run / 'inputs' / f'{iteration}.npz', allow_pickle=False) as inputs:
        assert inputs['num_graphs'] == 2
        assert inputs['time_step'].shape == (2,)
        assert inputs['pos_noise'].shape == inputs['pos'].shape
    return val_values


def test_qm9_native_checkpoint_resumes_both_adam_updates(tmp_path):
    asset_dir = os.environ.get('GEODIFF_QM9_SMOKE_ASSETS')
    if not asset_dir:
        pytest.skip('Set GEODIFF_QM9_SMOKE_ASSETS to the exported official smoke assets')
    assets = Path(asset_dir).resolve()
    weights_path = assets / 'qm9_default.npz'
    config = yaml.safe_load((ROOT / 'geodiff_paddle/configs/qm9_default.yml').read_text())
    config['dataset'].update(
        train=os.path.relpath(assets / 'train_2.npz', ROOT),
        val=os.path.relpath(assets / 'val_2.npz', ROOT),
    )
    config['train']['batch_size'] = 2
    smoke_config = tmp_path / 'qm9_smoke.yml'
    smoke_config.write_text(yaml.safe_dump(config))

    first_run = run_trainer(smoke_config, tmp_path / 'first',
                            '--weights', str(weights_path), '--max_iters', '1')
    resumed_run = run_trainer(first_run, tmp_path / 'resume',
                              '--resume_iter', '1', '--max_iters', '2')
    first_metrics, resume_metrics = read_metrics(first_run, 1), read_metrics(resumed_run, 2)

    import paddle

    first = paddle.load(str(first_run / 'checkpoints/1.pdckpt'))
    resumed = paddle.load(str(resumed_run / 'checkpoints/2.pdckpt'))
    assert {path.name for path in (first_run / 'checkpoints').iterdir()} == {'1.pdckpt'}
    assert {path.name for path in (resumed_run / 'checkpoints').iterdir()} == {'2.pdckpt'}
    for iteration, state, metrics in ((1, first, first_metrics), (2, resumed, resume_metrics)):
        assert {'model', 'config', 'iteration', 'avg_val_loss',
                'optimizer_global', 'optimizer_local'} <= state.keys()
        assert state['iteration'] == state['config']['train']['max_iters'] == iteration
        assert type(state['config']) is type(state['config']['train']) is dict
        assert state['avg_val_loss'] == metrics[0]

    with np.load(weights_path, allow_pickle=False) as weights:
        for branch, metric_index in (('global', 1), ('local', 2)):
            first_opt = first[f'optimizer_{branch}']
            resumed_opt = resumed[f'optimizer_{branch}']
            first_scheduler = first_opt['LR_Scheduler']
            resumed_scheduler = resumed_opt['LR_Scheduler']
            assert first_scheduler['best'] == first_metrics[metric_index]
            best, current = first_metrics[metric_index], resume_metrics[metric_index]
            assert resumed_scheduler['best'] == (current if current < best * (1 - 1e-4) else best)
            assert resumed_scheduler['last_epoch'] == first_scheduler['last_epoch'] + 1

            keys = [key for key in weights.files if key.startswith(f'model_{branch}.')]
            assert any(not np.array_equal(weights[key], first['model'][key].numpy()) for key in keys)
            assert any(not np.array_equal(first['model'][key].numpy(), resumed['model'][key].numpy())
                       for key in keys)

            for name in ('moment1', 'moment2', 'beta1_pow_acc', 'beta2_pow_acc'):
                keys = [key for key in first_opt if f'_{name}_' in key]
                assert keys, name
                before = [first_opt[key].numpy() for key in keys]
                after = [resumed_opt[key].numpy() for key in keys]
                if name.startswith('beta'):
                    beta = first['config']['train']['optimizer'][name.split('_')[0]]
                    for old, new in zip(before, after):
                        np.testing.assert_allclose(old, beta ** 2, rtol=1e-6)
                        np.testing.assert_allclose(new, old * beta, rtol=1e-6)
                else:
                    assert any(np.any(value) for value in before)
                    assert any(not np.array_equal(old, new) for old, new in zip(before, after))


@pytest.mark.parametrize(('kind', 'factor', 'milestone'), [
    ('expmin', 0.5, None), ('expmin_milestone', 0.25, 2),
])
def test_exponential_scheduler_decays_to_floor_and_restores(kind, factor, milestone, monkeypatch):
    import paddle

    monkeypatch.syspath_prepend(str(ROOT / 'geodiff_paddle'))
    from utils.common import get_optimizer, get_scheduler

    optimizer_cfg = SimpleNamespace(type='adam', lr=0.01, weight_decay=0.0, beta1=0.9, beta2=0.999)
    scheduler_cfg = SimpleNamespace(type=kind, factor=factor, min_lr=0.003, milestone=milestone)
    optimizer = get_optimizer(optimizer_cfg, paddle.nn.Linear(1, 1))
    scheduler = get_scheduler(scheduler_cfg, optimizer)
    assert optimizer.get_lr() == pytest.approx(0.01)
    scheduler.step()
    assert optimizer.get_lr() == pytest.approx(0.005)

    resumed = get_optimizer(optimizer_cfg, paddle.nn.Linear(1, 1))
    resumed_scheduler = get_scheduler(scheduler_cfg, resumed)
    resumed.set_state_dict(optimizer.state_dict())
    assert resumed.get_lr() == pytest.approx(0.005)
    for _ in range(2):
        resumed_scheduler.step()
        assert resumed.get_lr() == pytest.approx(0.003)
