"""Replay one recorded training step through the original Torch model.

--inputs is an NPZ written by train.py --record_inputs. The recorded time_step and
pos_noise are injected into get_loss, one forward/backward runs and no optimizer
steps. Gradients are saved as they come out of backward, before the gradient
clipping whose norm is saved as grad_norm.

Run from the repository root in the Torch/PyG environment:

    python tools/replay_torch_step.py --checkpoint checkpoints/qm9_default.pt \
        --inputs runs/torch_qm9_smoke/<run>/inputs/2865000.npz --out runs/l1_alignment/torch.npz

With --graphs-source and --graphs-out instead of --checkpoint and --out, the tool
exports the dataset graphs of the recorded batch from the official split:

    python tools/replay_torch_step.py --inputs runs/torch_qm9_smoke/<run>/inputs/2865000.npz \
        --graphs-source data/QM9/train_data_40k.pkl --graphs-out runs/l1_alignment/graphs.npz
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from export_paddle_assets import TORCH_ROOT, _CPUUnpickler


INPUTS = ('atom_type', 'pos', 'bond_index', 'bond_type', 'batch', 'num_nodes_per_graph')
EDGE_NAMES = ('edge_index', 'edge_type', 'edge_length', 'local_edge_mask')
GRAPH_ARRAYS = ('atom_type', 'pos', 'edge_index', 'edge_type')
GRAD = 'grad::'


def to_array(tensor):
    """Copy of a tensor as a NumPy array that no longer shares its storage."""
    return tensor.detach().cpu().numpy().copy()


def replay_step(model, inputs, time_step, pos_noise, num_graphs, max_norm=10000.0):
    """Run one replayed step and return the arrays of the result archive."""
    edges, used = {}, []
    # The loss forward is the only forward; a hook keeps its edges.
    handle = model.register_forward_hook(
        lambda module, args, output: edges.update(zip(EDGE_NAMES, output[2:])))
    try:
        loss, loss_global, loss_local = model.get_loss(
            replay=(time_step, pos_noise), record_inputs=lambda *tensors: used.append(tensors),
            return_unreduced_loss=True, **inputs, num_graphs=num_graphs)
    finally:
        handle.remove()
    assert all(torch.equal(a, b) for a, b in zip(used[0], (time_step, pos_noise))), \
        'the loss did not use the recorded time_step and pos_noise'
    loss.mean().backward()

    out = dict(
        loss=to_array(loss), loss_global=to_array(loss_global), loss_local=to_array(loss_local),
        loss_mean=np.float64(loss.mean().item()),
        **{name: to_array(value) for name, value in edges.items()},
        num_nodes=np.int64(inputs['pos'].shape[0]),
    )
    # named_parameters() lists shared tensors once, under their first name.
    parameters = dict(model.named_parameters())
    no_grad = []
    for name, param in parameters.items():
        # betas and alphas are fixed buffers; only trainable parameters count.
        if not param.requires_grad:
            continue
        if param.grad is None:
            no_grad.append(name)
        else:
            # Copied before the in-place clipping below changes the gradients.
            out[GRAD + name] = to_array(param.grad)
    grad_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), max_norm=max_norm)
    out.update(grad_norm=np.float64(grad_norm.item()), no_grad=np.array(no_grad))
    return out


def graph_key(atom_type, pos, edge_index, edge_type):
    return b''.join(array.tobytes() for array in (atom_type, pos, edge_index, edge_type))


def split_batch(recorded):
    """Per-graph slices of the recorded atoms and of the molecule-local bond arrays."""
    counts = recorded['num_nodes_per_graph']
    node_ptr = np.concatenate([[0], counts.cumsum()])
    offset_bond_index = recorded['bond_index']
    graph_of_bond = np.searchsorted(node_ptr, offset_bond_index[0], side='right') - 1
    assert np.all(np.diff(graph_of_bond) >= 0), 'bonds must be grouped by graph'
    bond_ptr = np.concatenate([[0], np.bincount(graph_of_bond, minlength=len(counts)).cumsum()])
    # The batch offsets every bond by its graph's first node; drop that offset.
    bond_index = offset_bond_index - node_ptr[graph_of_bond]
    nodes = [slice(node_ptr[i], node_ptr[i + 1]) for i in range(len(counts))]
    bonds = [slice(bond_ptr[i], bond_ptr[i + 1]) for i in range(len(counts))]
    return {
        'atom_type': [recorded['atom_type'][s] for s in nodes],
        'pos': [recorded['pos'][s] for s in nodes],
        'edge_index': [bond_index[:, s] for s in bonds],
        'edge_type': [recorded['bond_type'][s] for s in bonds],
    }, node_ptr, bond_ptr


def official_graphs(source, recorded):
    """Arrays of the dataset graphs that make up the recorded batch, in batch order.

    Each recorded graph is matched to a dataset graph by its atom types,
    positions, bond indices and bond types. Atoms, bonds and types are then
    asserted to be equal one by one, which fixes their order to the recording.
    """
    with source.open('rb') as file:
        graphs = _CPUUnpickler(file).load()
    index = {}
    for position, graph in enumerate(graphs):
        key = graph_key(*(getattr(graph, name).numpy() for name in GRAPH_ARRAYS))
        index.setdefault(key, []).append(position)

    batch, node_ptr, bond_ptr = split_batch(recorded)
    out = {name: [] for name in GRAPH_ARRAYS}
    positions = []
    for i in range(len(batch['atom_type'])):
        key = graph_key(*(batch[name][i] for name in GRAPH_ARRAYS))
        if key not in index:
            raise ValueError(f'Recorded graph {i} is not in {source}')
        graph = graphs[index[key][0]]
        positions.append(index[key][0])
        for name in GRAPH_ARRAYS:
            array = getattr(graph, name).numpy()
            if not np.array_equal(array, batch[name][i]):
                raise ValueError(f'Recorded graph {i}: {name} differ from the dataset graph')
            out[name].append(array)
    return dict(
        atom_type=np.concatenate(out['atom_type']), pos=np.concatenate(out['pos']),
        edge_index=np.concatenate(out['edge_index'], axis=1), edge_type=np.concatenate(out['edge_type']),
        node_ptr=node_ptr, edge_ptr=bond_ptr, dataset_index=np.asarray(positions, dtype=np.int64),
        time_step=recorded['time_step'], pos_noise=recorded['pos_noise'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='trusted GeoDiff Torch checkpoint')
    parser.add_argument('--inputs', type=Path, required=True, help='step NPZ from train.py --record_inputs')
    parser.add_argument('--out', type=Path, help='result NPZ of the replayed step')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--graphs-source', type=Path, help='official training-split pickle')
    parser.add_argument('--graphs-out', type=Path,
                        help='graph_data-layout NPZ of the recorded batch from the official split')
    args = parser.parse_args()
    if (args.checkpoint is None) != (args.out is None) or (
            args.graphs_source is None) != (args.graphs_out is None):
        parser.error('--out needs --checkpoint, --graphs-out needs --graphs-source')
    if args.out is None and args.graphs_out is None:
        parser.error('nothing to write: give --out or --graphs-out')

    with np.load(args.inputs, allow_pickle=False) as step:
        recorded = {name: step[name] for name in step.files}
    if args.graphs_out is not None:
        graphs = official_graphs(args.graphs_source, recorded)
        args.graphs_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.graphs_out, **graphs)
        print(f'{args.graphs_out}: {len(graphs["dataset_index"])} graphs written from '
              f'{args.graphs_source}, dataset positions {graphs["dataset_index"].min()}'
              f'..{graphs["dataset_index"].max()}')
    if args.out is None:
        return

    sys.path.insert(0, str(TORCH_ROOT))
    from models.epsnet import get_model

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = get_model(checkpoint['config'].model).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.train()

    inputs = {key: torch.from_numpy(recorded[key]).to(args.device) for key in INPUTS}
    time_step = torch.from_numpy(recorded['time_step']).to(args.device)
    pos_noise = torch.from_numpy(recorded['pos_noise']).to(args.device)
    out = replay_step(model, inputs, time_step, pos_noise, int(recorded['num_graphs']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    no_grad = [str(name) for name in out['no_grad']]
    print(f'Replayed {int(recorded["num_graphs"])} graphs ({out["num_nodes"]} nodes, '
          f'{out["edge_type"].shape[0]} edges)')
    print(f'Loss {out["loss_mean"]:.17g} | Loss(Global) {out["loss_global"].mean():.17g} | '
          f'Loss(Local) {out["loss_local"].mean():.17g} | Grad {out["grad_norm"]:.17g}')
    print(f'{sum(name.startswith(GRAD) for name in out)} gradients saved, '
          f'{len(no_grad)} without gradient: {no_grad}')


if __name__ == '__main__':
    main()
