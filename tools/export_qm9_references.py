"""Recreate QM9 reference archives in the separate Torch/PyG CUDA environment.

Only supply trusted released checkpoints and preprocessed PyG pickles.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from export_paddle_assets import TORCH_ROOT, _GraphPrefixUnpickler, _array


def fixed_forward(checkpoint):
    from models.epsnet import get_model

    config = checkpoint['config'].model
    model = get_model(config).cuda().eval()
    model.load_state_dict(checkpoint['model'])

    # Reproduce the original fixed-forward run's exact NumPy inputs.
    seed = 2021
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 0.6, (9, 3)).astype(np.float32)
    pos[:, 0] += 1.2 * np.array([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=np.float32)
    bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]], dtype=np.int64)
    bond_index = np.concatenate((bonds, bonds[:, ::-1])).T
    inputs = {
        'atom_type': np.array([6, 6, 6, 6, 8, 6, 6, 6, 8], dtype=np.int64),
        'pos': pos,
        'bond_index': bond_index,
        'bond_type': np.ones(bond_index.shape[1], dtype=np.int64),
        'batch': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        'time_step': rng.integers(0, config.num_diffusion_timesteps, size=2, dtype=np.int64),
    }
    with torch.no_grad():
        outputs = model(**{key: torch.from_numpy(x).cuda() for key, x in inputs.items()}, return_edges=True)
    names = ('edge_inv_global', 'edge_inv_local', 'edge_index', 'edge_type', 'edge_length', 'local_edge_mask')
    return dict(seed=seed, iteration=checkpoint['iteration'], **inputs,
                **{key: _array(value) for key, value in zip(names, outputs, strict=True)})


def first_val_graph(path, config):
    from models.common import extend_graph_order_radius
    from torch_geometric.data import Batch

    with path.open('rb') as file:
        graph, = _GraphPrefixUnpickler(file, 1).load()
    batch = Batch.from_data_list([graph]).cuda()
    with torch.no_grad():
        edge_index, edge_type = extend_graph_order_radius(
            batch.num_nodes, batch.pos, batch.edge_index, batch.edge_type, batch.batch,
            order=config.edge_order, cutoff=config.cutoff,
        )
    return dict(atom_type=_array(batch.atom_type), pos=_array(batch.pos), batch=_array(batch.batch),
                bond_index=_array(batch.edge_index), bond_type=_array(batch.edge_type),
                edge_index=_array(edge_index), edge_type=_array(edge_type))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--val', type=Path, help='validation pickle (requires --val-graph)')
    parser.add_argument('--forward', type=Path, required=True, help='new fixed-forward archive')
    parser.add_argument('--val-graph', type=Path, help='new validation graph archive (requires --val)')
    args = parser.parse_args()
    if (args.val is None) != (args.val_graph is None):
        parser.error('--val and --val-graph must be supplied together')
    if args.val_graph is not None and args.forward.resolve() == args.val_graph.resolve():
        parser.error('Specify two different output paths')
    for path in (args.forward, args.val_graph):
        if path is not None and path.exists():
            parser.error(f'Output already exists: {path}')

    sys.path.insert(0, str(TORCH_ROOT))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    outputs = [(args.forward, fixed_forward(checkpoint))]
    if args.val_graph is not None:
        outputs.append((args.val_graph, first_val_graph(args.val, checkpoint['config'].model)))
    for path, arrays in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as file:
            np.savez(file, **arrays)


if __name__ == '__main__':
    main()
