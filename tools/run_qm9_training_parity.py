"""Prepare, run, or compare the fixed-input QM9 twenty-update experiment.

Run Torch and Paddle in separate processes with their native environments.
Prepare and compare need only NumPy. Output files must not already exist.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STEPS = 20
TIME_LEVELS = 5000
BATCH_FIELDS = ('atom_type', 'pos', 'bond_index', 'bond_type', 'batch', 'num_nodes_per_graph')
LOSS_FIELDS = ('loss', 'loss_global', 'loss_local')
STEP_FIELDS = (*LOSS_FIELDS, 'grad_norm', 'lr_global', 'lr_local')


def load_inputs(path, taped):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in (*BATCH_FIELDS, 'num_graphs', 'time_step', 'pos_noise')}
    for name, array in arrays.items():
        dtype = np.float32 if name in ('pos', 'pos_noise') else np.int64
        if array.dtype != dtype:
            raise ValueError(f'{name}: expected {np.dtype(dtype)}, got {array.dtype}')
    if arrays['num_graphs'].shape != () or arrays['num_graphs'] <= 0:
        raise ValueError('num_graphs must be a positive scalar')
    graphs, nodes, bonds = (arrays[name].size for name in ('num_nodes_per_graph', 'atom_type', 'bond_type'))
    prefix = (STEPS,) if taped else ()
    shapes = dict(num_nodes_per_graph=(int(arrays['num_graphs']),), atom_type=(nodes,),
                  pos=(nodes, 3), bond_index=(2, bonds), bond_type=(bonds,), batch=(nodes,),
                  time_step=(*prefix, graphs), pos_noise=(*prefix, nodes, 3))
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f'{name}: expected shape {shape}, got {arrays[name].shape}')
    if ((arrays['time_step'] < 0) | (arrays['time_step'] >= TIME_LEVELS)).any():
        raise ValueError(f'time_step must be in [0, {TIME_LEVELS})')
    return arrays


def prepare(inputs, out):
    arrays = load_inputs(inputs, taped=False)
    original_t, original_noise = arrays['time_step'], arrays['pos_noise']
    times = np.empty((STEPS, *original_t.shape), dtype=np.int64)
    noises = np.empty((STEPS, *original_noise.shape), dtype=np.float32)
    times[0], noises[0] = original_t, original_noise
    rng = np.random.default_rng(2021)
    graphs = original_t.size
    for i in range(1, STEPS):
        # Match GeoDiff's order: draws first, then their complements, truncated to the batch size.
        draws = rng.integers(0, TIME_LEVELS, size=graphs // 2 + 1, dtype=np.int64)
        times[i] = np.concatenate((draws, TIME_LEVELS - 1 - draws))[:graphs]
        rng.standard_normal(original_noise.shape, dtype=np.float32, out=noises[i])
    arrays.update(time_step=times, pos_noise=noises)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as file:
        np.savez(file, **arrays)


def run_training(args):
    arrays = load_inputs(args.inputs, taped=True)
    is_torch = args.command == 'torch'
    backend_root = ROOT / ('geodiff' if is_torch else 'geodiff_paddle')
    sys.path.insert(0, str(backend_root))

    import yaml
    from easydict import EasyDict
    from models.epsnet import get_model
    from utils.common import get_optimizer

    with (backend_root / 'configs/qm9_default.yml').open() as file:
        config = EasyDict(yaml.safe_load(file))

    if is_torch:
        import torch
        from torch.nn.utils import clip_grad_norm_

        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = get_model(checkpoint['config'].model).to(args.device)
        model.load_state_dict(checkpoint['model'])
        del checkpoint

        def tensor(array):
            return torch.from_numpy(array).to(args.device)

        def clear_grad(optimizer):
            optimizer.zero_grad(set_to_none=True)

        def learning_rate(optimizer):
            return optimizer.param_groups[0]['lr']
    else:
        import paddle
        from paddle.nn.utils import clip_grad_norm_
        from models.weights import load_released_weights

        paddle.set_device(args.device)
        with paddle.utils.unique_name.guard():
            model = get_model(config.model)
        load_released_weights(model, args.weights)

        def tensor(array):
            return paddle.to_tensor(array, place=args.device)

        def clear_grad(optimizer):
            optimizer.clear_grad(set_to_zero=False)

        def learning_rate(optimizer):
            return optimizer.get_lr()

    model.train()
    train = config.train
    global_adam = get_optimizer(train.optimizer, model.model_global)
    local_adam = get_optimizer(train.optimizer, model.model_local)
    clip_parameters = list({id(param): param for param in model.parameters()}.values())
    inputs = {name: tensor(arrays[name]) for name in BATCH_FIELDS}
    times, noises = tensor(arrays['time_step']), tensor(arrays['pos_noise'])
    num_graphs = int(arrays['num_graphs'])
    rows = []
    for i in range(STEPS):
        clear_grad(global_adam)
        clear_grad(local_adam)
        loss, loss_global, loss_local = model.get_loss(
            **inputs, num_graphs=num_graphs, anneal_power=train.anneal_power,
            replay=(times[i], noises[i]), return_unreduced_loss=True,
        )
        loss = loss.mean()
        loss.backward()
        grad_norm = clip_grad_norm_(clip_parameters, max_norm=train.max_grad_norm)
        row = dict(step=i + 1, loss=loss.item(), loss_global=loss_global.detach().mean().item(),
                   loss_local=loss_local.detach().mean().item(), grad_norm=grad_norm.item(),
                   lr_global=learning_rate(global_adam), lr_local=learning_rate(local_adam))
        global_adam.step()
        local_adam.step()
        rows.append(row)
        print(f'{args.command} step {i + 1:02d} | '
              + ' | '.join(f'{name}={row[name]:.17g}' for name in STEP_FIELDS), flush=True)
    write_json(args.out, rows)


def read_steps(path, side):
    with path.open() as file:
        rows = json.load(file)
    if not isinstance(rows, list) or len(rows) != STEPS:
        raise ValueError(f'{side}: expected exactly {STEPS} step records')
    for step, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] != step:
            raise ValueError(f'{side}: expected step {step} in order')
        for name in STEP_FIELDS:
            value = row.get(name)
            if type(value) not in (int, float) or not np.isfinite(float(value)):
                raise ValueError(f'{side} step {step}: {name} must be a finite number')
    return rows


def compare(torch_path, paddle_path):
    torch, paddle = read_steps(torch_path, 'torch'), read_steps(paddle_path, 'paddle')
    reference, actual = (np.array([[row[name] for name in LOSS_FIELDS] for row in source], dtype=np.float64)
                         for source in (torch, paddle))
    delta = actual - reference
    absolute = np.abs(delta)
    relative = absolute / np.maximum(np.maximum(np.abs(actual), np.abs(reference)), 1e-30)
    metrics = {}
    for i, name in enumerate(LOSS_FIELDS):
        d = delta[:, i]
        metrics[name] = dict(mean_abs=float(absolute[:, i].mean()), mean_rel=float(relative[:, i].mean()),
                             min_diff=float(d.min()), max_diff=float(d.max()),
                             positive=int((d > 0).sum()), negative=int((d < 0).sum()), zero=int((d == 0).sum()))
    steps = [dict(step=i + 1,
                  torch={name: torch[i][name] for name in STEP_FIELDS},
                  paddle={name: paddle[i][name] for name in STEP_FIELDS},
                  delta=dict(zip(LOSS_FIELDS, delta[i].tolist())),
                  abs_error=dict(zip(LOSS_FIELDS, absolute[i].tolist())),
                  rel_error=dict(zip(LOSS_FIELDS, relative[i].tolist()))) for i in range(STEPS)]
    passed = metrics['loss']['min_diff'] < 0 < metrics['loss']['max_diff']
    return dict(passed=passed, steps=steps, metrics=metrics)


def write_json(path, data):
    content = json.dumps(data, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        file.write(content + '\n')


def print_comparison(report):
    for row in report['steps']:
        print(f'Step {row["step"]:02d} | ' + ' | '.join(
            f'{name}: {row["torch"][name]:.17g} -> {row["paddle"][name]:.17g} '
            f'(Paddle-Torch={row["delta"][name]:+.17g})' for name in LOSS_FIELDS))
    for name, m in report['metrics'].items():
        print(f'{name}: MAE={m["mean_abs"]:.17g}, MRE={m["mean_rel"]:.17g}, '
              f'd=[{m["min_diff"]:+.17g}, {m["max_diff"]:+.17g}], '
              f'+/-/0={m["positive"]}/{m["negative"]}/{m["zero"]}')
    print(f'{"PASS" if report["passed"] else "FAIL"}: total loss cross-zero criterion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='build the shared random tape from a recorded flat batch')
    torch = commands.add_parser('torch', help='train for twenty updates in the Torch environment')
    paddle = commands.add_parser('paddle', help='train for twenty updates in the Paddle environment')
    comparison = commands.add_parser('compare', help='compare backend step JSON files using NumPy')
    for command in (prep, torch, paddle):
        command.add_argument('--inputs', type=Path, required=True)
    for command in (prep, torch, paddle, comparison):
        command.add_argument('--out', type=Path, required=True, help='new output file')
    torch.add_argument('--checkpoint', type=Path, required=True, help='trusted Torch checkpoint')
    torch.add_argument('--device', default='cuda:0')
    paddle.add_argument('--weights', type=Path, required=True, help='native Paddle .pdparams')
    paddle.add_argument('--device', default='gpu:0')
    comparison.add_argument('--torch', type=Path, required=True)
    comparison.add_argument('--paddle', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f'output already exists: {args.out}')
        if args.command == 'prepare':
            prepare(args.inputs, args.out)
        elif args.command == 'compare':
            report = compare(args.torch, args.paddle)
            write_json(args.out, report)
            print_comparison(report)
            return 0 if report['passed'] else 1
        else:
            run_training(args)
    except (OSError, ValueError, KeyError, OverflowError) as error:
        parser.exit(2, f'{args.command}: {error}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
