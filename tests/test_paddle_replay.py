from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from models.epsnet import get_model
from tools.replay_paddle_step import replay_step

def build_model(model_type='diffusion'):
    """Tiny dual encoder in train mode that counts and keeps its forwards."""

    config = SimpleNamespace(
        type=model_type, network='dualenc', hidden_dim=4, num_convs=2, num_convs_local=1,
        cutoff=2.0, mlp_act='relu', beta_schedule='sigmoid', beta_start=1e-7, beta_end=2e-3,
        num_diffusion_timesteps=10, sigma_begin=0.01, sigma_end=1.0, num_noise_level=10,
        edge_order=2, edge_encoder='mlp', smooth_conv=False,
    )

    class Capturing(type(get_model(config))):
        def forward(self, *args, **kwargs):
            self.forward_calls = getattr(self, 'forward_calls', 0) + 1
            self.seen_pos = kwargs['pos']
            output = super().forward(*args, **kwargs)
            self.seen_edges = output[2:]
            return output

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


def inputs():
    """The batch without the graph count, which replay_step takes separately."""
    return {key: value for key, value in batch().items() if key != 'num_graphs'}


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


def test_replay_runs_one_forward_and_saves_the_edges_of_that_forward():
    model = build_model()
    replay = (paddle.to_tensor([2, 7]), paddle.full([5, 3], 0.5))
    result = replay_step(model, inputs(), *replay, num_graphs=2)

    assert model.forward_calls == 1
    for name, edges in zip(('edge_index', 'edge_type', 'edge_length', 'local_edge_mask'), model.seen_edges):
        np.testing.assert_array_equal(result[name], edges.numpy(), err_msg=name)


def test_replay_saves_untouched_gradients_with_loss_mean_and_pre_clip_norm():
    model = build_model()
    replay = (paddle.to_tensor([2, 7]), paddle.full([5, 3], 0.5))
    # Any real step is above this bound, so clipping would scale every gradient.
    result = replay_step(model, inputs(), *replay, num_graphs=2, max_norm=1e-6)

    saved_norm = np.sqrt(sum(np.square(value.astype(np.float64)).sum()
                             for name, value in result.items() if name.startswith('grad::')))
    assert saved_norm > 1e-6
    np.testing.assert_allclose(result['grad_norm'], saved_norm, rtol=1e-5)
    np.testing.assert_allclose(result['loss_mean'], result['loss'].astype(np.float64).mean(), rtol=1e-6)


def test_replay_is_rejected_for_the_dsm_loss():
    model = build_model(model_type='dsm')
    with pytest.raises(ValueError, match='replay'):
        model.get_loss(replay=(paddle.to_tensor([2, 7]), paddle.full([5, 3], 0.5)), **batch())
