from types import SimpleNamespace

import numpy as np
import paddle

from models.epsnet import get_model


def build_model():
    """Tiny dual encoder in train mode that keeps the positions its forward gets."""

    config = SimpleNamespace(
        type='diffusion', network='dualenc', hidden_dim=4, num_convs=2, num_convs_local=1,
        cutoff=2.0, mlp_act='relu', beta_schedule='sigmoid', beta_start=1e-7, beta_end=2e-3,
        num_diffusion_timesteps=10, edge_order=2, edge_encoder='mlp', smooth_conv=False,
    )

    class Capturing(type(get_model(config))):
        def forward(self, *args, **kwargs):
            self.seen_pos = kwargs['pos']
            return super().forward(*args, **kwargs)

    return Capturing(config)


def batch():
    # A chain of three bonds and a lone atom, in two graphs.
    return dict(
        atom_type=paddle.to_tensor([1, 2, 3, 4, 5]),
        pos=paddle.to_tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 0., 1.], [9., 9., 9.]]),
        bond_index=paddle.to_tensor([[0, 1, 2], [1, 2, 3]]),
        bond_type=paddle.to_tensor([1, 2, 3]),
        batch=paddle.to_tensor([0, 0, 0, 0, 1]),
        num_nodes_per_graph=paddle.to_tensor([4, 1]),
        num_graphs=2,
    )


def test_replay_perturbs_positions_with_injected_time_and_noise():
    model = build_model()
    alphas = model.alphas.numpy()
    replay = (paddle.to_tensor([2, 7]), paddle.full([5, 3], 0.5))
    model.get_loss(replay=replay, **batch())

    a_pos = alphas[[2, 2, 2, 2, 7], None]
    expected = batch()['pos'].numpy() + 0.5 * (1 - a_pos) ** 0.5 / a_pos ** 0.5
    np.testing.assert_allclose(model.seen_pos.numpy(), expected, rtol=1e-6)


def test_replay_makes_loss_reproducible_and_records_inputs():
    model = build_model()
    replay = (paddle.to_tensor([2, 7]), paddle.full([5, 3], 0.5))
    seen = []
    first = model.get_loss(replay=replay, record_inputs=lambda *tensors: seen.append(tensors), **batch())
    second = model.get_loss(replay=replay, **batch())
    assert first.shape == [5, 1] and paddle.isfinite(first).all().item()
    np.testing.assert_array_equal(first.numpy(), second.numpy())
    np.testing.assert_array_equal(seen[0][0].numpy(), replay[0].numpy())
    np.testing.assert_array_equal(seen[0][1].numpy(), replay[1].numpy())
