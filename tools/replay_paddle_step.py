"""Replay one recorded training step through Paddle and save loss and gradients.

--inputs is an NPZ written by train.py --record_inputs. The recorded time_step and
pos_noise are injected, one forward/backward runs and no optimizer steps.
Gradients are saved as they come out of backward, before the gradient clipping
whose norm is saved as grad_norm. --graphs-out only repacks the recorded arrays
in the layout read by graph_data.load_graphs; it does not export the dataset.
Run with PYTHONPATH=geodiff_paddle from the repository root.
"""

import argparse
from pathlib import Path

import numpy as np
import paddle
import yaml
from easydict import EasyDict

from models.epsnet import get_model
from models.weights import load_released_weights


INPUTS = ('atom_type', 'pos', 'bond_index', 'bond_type', 'batch', 'num_nodes_per_graph')
EDGE_NAMES = ('edge_index', 'edge_type', 'edge_length', 'local_edge_mask')
GRAD = 'grad::'


def dedup_parameters(model):
    """Yield each physical parameter once, under its first registered name."""
    seen = set()
    for name, param in model.named_parameters():
        if id(param) not in seen:
            seen.add(id(param))
            yield name, param


def linear_weight_ids(model):
    """Ids of Linear weight parameters, which Paddle stores as [in, out]."""
    return {id(param) for layer in model.sublayers() if isinstance(layer, paddle.nn.Linear)
            for name, param in layer.named_parameters(include_sublayers=False) if name.endswith('weight')}


def save_graphs(path, inputs, time_step, pos_noise):
    """Write the batch in the layout read by graph_data.load_graphs."""
    cpu = {key: value.numpy() for key, value in inputs.items()}
    counts, pos = cpu['num_nodes_per_graph'], cpu['pos']
    (edge_index, edge_type) = (cpu['bond_index'], cpu['bond_type'])
    node_ptr = np.concatenate([[0], counts.cumsum()])
    graph_of_edge = np.searchsorted(node_ptr, edge_index[0], side='right') - 1
    edge_ptr = np.concatenate([[0], np.bincount(graph_of_edge, minlength=len(counts)).cumsum()])
    local_index = edge_index - node_ptr[graph_of_edge]
    assert np.all(np.diff(graph_of_edge) >= 0), 'bonds must be grouped by graph'
    np.savez(path, atom_type=cpu['atom_type'], pos=pos, edge_index=local_index,
             edge_type=edge_type, node_ptr=node_ptr, edge_ptr=edge_ptr,
             time_step=time_step, pos_noise=pos_noise)


def export_gradients(parameters, transposed):
    """Return grad:: arrays in Torch layout and the names without a gradient."""
    out = {}
    for name, param in parameters.items():
        if param.grad is not None:
            # Linear weight gradients are saved transposed, in the [out, in]
            # layout Torch records; every other array keeps its Paddle layout.
            grad = param.grad.T if id(param) in transposed else param.grad
            out[GRAD + name] = grad.numpy()
    return out, [name for name, param in parameters.items() if param.grad is None]


def replay_step(model, inputs, time_step, pos_noise, num_graphs, max_norm=10000.0):
    """Run one replayed step and return the arrays of the result archive."""
    edges = {}
    # The loss forward is the only forward; a post hook keeps its edges.
    hook = model.register_forward_post_hook(
        lambda layer, args, output: edges.update(zip(EDGE_NAMES, output[2:])))
    try:
        (loss, loss_global, loss_local) = model.get_loss(
            replay=(time_step, pos_noise), return_unreduced_loss=True, **inputs, num_graphs=num_graphs)
    finally:
        hook.remove()
    loss.mean().backward()
    parameters = dict(dedup_parameters(model))
    out = dict(
        loss=loss.numpy(), loss_global=loss_global.numpy(), loss_local=loss_local.numpy(),
        loss_mean=np.float64(loss.mean().item()),
        **{name: value.numpy() for name, value in edges.items()},
        num_nodes=np.int64(inputs['pos'].shape[0]),
    )
    gradients, no_grad = export_gradients(parameters, linear_weight_ids(model))
    grad_norm = paddle.nn.utils.clip_grad_norm_(list(parameters.values()), max_norm=max_norm)
    out.update(gradients, grad_norm=np.float64(grad_norm.item()), no_grad=np.array(no_grad))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True, help='native Paddle .pdparams')
    parser.add_argument('--inputs', type=Path, required=True, help='step NPZ from train.py --record_inputs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--graphs-out', type=Path,
                        help='optional repack of the recorded arrays as a graph_data-layout NPZ')
    parser.add_argument('--device', default='gpu:0')
    args = parser.parse_args()
    paddle.set_device(args.device)

    config_path = Path(__file__).resolve().parents[1] / 'geodiff_paddle/configs/qm9_default.yml'
    with config_path.open() as config_file:
        config = EasyDict(yaml.safe_load(config_file)).model
    with paddle.utils.unique_name.guard():
        model = get_model(config)
    load_released_weights(model, args.weights)
    model.train()

    with np.load(args.inputs, allow_pickle=False) as step:
        inputs = {key: paddle.to_tensor(step[key], place=args.device) for key in INPUTS}
        time_step = paddle.to_tensor(step['time_step'], place=args.device)
        pos_noise = paddle.to_tensor(step['pos_noise'], place=args.device)
        num_graphs = int(step['num_graphs'])

    out = replay_step(model, inputs, time_step, pos_noise, num_graphs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    if args.graphs_out is not None:
        args.graphs_out.parent.mkdir(parents=True, exist_ok=True)
        save_graphs(args.graphs_out, inputs, time_step.numpy(), pos_noise.numpy())
    print(f'Replayed {num_graphs} graphs ({out["num_nodes"]} nodes, {out["edge_type"].shape[0]} edges)')
    print(f'Loss {out["loss_mean"]:.17g} | Loss(Global) {out["loss_global"].mean():.17g} | '
          f'Loss(Local) {out["loss_local"].mean():.17g} | Grad {out["grad_norm"]:.17g}')
    no_grad = [str(name) for name in out['no_grad']]
    print(f'{sum(name.startswith(GRAD) for name in out)} gradients saved, '
          f'{len(no_grad)} without gradient: {no_grad}')


if __name__ == '__main__':
    main()
