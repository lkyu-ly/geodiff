"""Replay released QM9 weights and reference NPZs with Paddle only.

--weights is the native .pdparams written by tools/convert_released_weights.py.
Run with PYTHONPATH=geodiff_paddle from the repository root.
"""

import argparse
from pathlib import Path

import numpy as np
import paddle
import yaml
from easydict import EasyDict

from graph_data import GraphDataLoader, load_graphs
from models.common import extend_graph_order_radius
from models.epsnet import get_model
from models.weights import load_released_weights


INPUTS = ('atom_type', 'pos', 'bond_index', 'bond_type', 'batch')
OUTPUTS = ('edge_inv_global', 'edge_inv_local', 'edge_index', 'edge_type', 'edge_length', 'local_edge_mask')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True, help='native Paddle .pdparams')
    parser.add_argument('--val-graph', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--device', default='gpu:0')
    args = parser.parse_args()
    paddle.set_device(args.device)
    paddle.seed(2021)
    print(f'Paddle {paddle.__version__} on {paddle.device.get_device()}')
    config_path = Path(__file__).resolve().parents[1] / 'geodiff_paddle/configs/qm9_default.yml'
    with config_path.open() as config_file:
        config = EasyDict(yaml.safe_load(config_file)).model

    model = get_model(config)
    load_released_weights(model, args.weights)
    state = model.state_dict()
    assert len(state) == 198 and len({id(tensor) for tensor in state.values()}) == 100
    weights = paddle.load(str(args.weights))
    for key, value in state.items():
        np.testing.assert_array_equal(value.numpy(), weights[key].numpy(), err_msg=key)
    print('Loaded 198 exact state entries (98 shared aliases)')
    model.eval()
    with paddle.no_grad(), np.load(args.forward, allow_pickle=False) as ref:
        actual = model(*(paddle.to_tensor(ref[key]) for key in (*INPUTS, 'time_step')), return_edges=True)
        for key, value in zip(OUTPUTS, actual):
            value = value.numpy()
            expected = ref[key]
            assert value.shape == expected.shape and value.dtype == expected.dtype, key
            if key in ('edge_inv_global', 'edge_inv_local', 'edge_length'):
                np.testing.assert_allclose(value, expected, atol=1e-5, rtol=1e-4, err_msg=key)
                error = np.abs(value - expected)
                print(f'{key} {value.shape}: max_abs={error.max():.8g} mean_abs={error.mean():.8g}')
            else:
                np.testing.assert_array_equal(value, expected, err_msg=key)
                print(f'{key} {value.shape}: exact')

    with np.load(args.val_graph, allow_pickle=False) as ref:
        pos, bond_index, bond_type, batch = (paddle.to_tensor(ref[key]) for key in INPUTS[1:])
        edge_index, edge_type = extend_graph_order_radius(
            pos.shape[0], pos, bond_index, bond_type, batch, order=config.edge_order, cutoff=config.cutoff
        )
        assert edge_index.shape == [2, 306] and edge_type.shape == [306]
        np.testing.assert_array_equal(edge_index.numpy(), ref['edge_index'])
        np.testing.assert_array_equal(edge_type.numpy(), ref['edge_type'])
        print(f'First official val graph: {pos.shape[0]} nodes, 306 edges and types exact')

    # A fresh model keeps eval-time embedding renormalization out of the loss check.
    model = get_model(config)
    load_released_weights(model, args.weights)
    graphs = load_graphs(args.train)
    batch = next(iter(GraphDataLoader(graphs, batch_size=len(graphs), shuffle=False)))
    loss, loss_global, loss_local = model.get_loss(
        batch.atom_type, batch.pos, batch.edge_index, batch.edge_type, batch.batch,
        batch.ptr[1:] - batch.ptr[:-1], batch.num_graphs, return_unreduced_loss=True,
    )
    for result in (loss, loss_global, loss_local):
        assert result.shape == [batch.pos.shape[0], 1] and paddle.isfinite(result).all().item()
    np.testing.assert_allclose(loss.numpy(), (loss_global + loss_local).numpy())
    loss.mean().backward()
    checked = (
        model.edge_encoder_global, model.encoder_global, model.grad_global_dist_mlp,
        model.encoder_local, model.grad_local_dist_mlp,
    )
    for layer in checked:
        for param in layer.parameters():
            assert param.grad is not None and paddle.isfinite(param.grad).all().item(), param.name
    assert all(param.grad is None for param in model.edge_encoder_local.parameters())
    assert model.betas.stop_gradient and model.alphas.stop_gradient
    print(f'Loss/backward over {batch.num_graphs} graphs ({batch.pos.shape[0]} nodes): '
          f'loss={loss.mean().item():.8g} global={loss_global.mean().item():.8g} local={loss_local.mean().item():.8g}')


if __name__ == '__main__':
    main()
