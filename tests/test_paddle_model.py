import numpy as np
import paddle
import pytest
from paddle import nn


def test_higher_order_edges_aligns_hops_with_zero_and_duplicate_types():
    from models.common import higher_order_edges
    from utils.chem import BOND_TYPES

    edge_index = paddle.to_tensor([[2, 1, 0, 0, 1], [1, 2, 1, 1, 0]])
    edge_type = paddle.to_tensor([3, 3, 0, 0, 2])
    actual_index, actual_type, actual_hop = higher_order_edges(5, edge_index, edge_type)
    np.testing.assert_array_equal(actual_index.numpy(), [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    np.testing.assert_array_equal(actual_type.numpy(), [0, len(BOND_TYPES) + 1, 2, 3, len(BOND_TYPES) + 1, 3])
    np.testing.assert_array_equal(actual_hop.numpy(), [1, 2, 1, 1, 2, 1])


def test_radius_union_keeps_zero_types_in_sorted_coordinates():
    from models.common import extend_graph_order_radius

    pos = paddle.to_tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
    edge_index = paddle.to_tensor([[2, 0, 0], [1, 2, 2]])
    edge_type = paddle.to_tensor([4, 2, 3])
    batch = paddle.to_tensor([0, 0, 0, 1])
    actual_index, actual_type = extend_graph_order_radius(
        len(pos), pos, edge_index, edge_type, batch, cutoff=1.5, extend_order=False
    )
    np.testing.assert_array_equal(actual_index.numpy(), [[0, 0, 1, 1, 2], [1, 2, 0, 2, 1]])
    np.testing.assert_array_equal(actual_type.numpy(), [0, 5, 0, 0, 4])


def test_gine_dense_and_sparse_messages_reach_targets_with_isolates():
    from models.encoder.gin import GINEConv

    # Negative messages exercise the activation before target reduction.
    mlp = nn.Linear(2, 1, bias_attr=False)
    mlp.weight.set_value(np.array([[2.], [3.]], dtype=np.float32))
    conv = GINEConv(mlp, eps=0.5, activation='relu')
    x_src = paddle.to_tensor([[1., -2.], [3., 4.]])
    x_dst = paddle.to_tensor([[1., 1.], [0., 0.], [2., 2.]])
    edge_attr = paddle.to_tensor([[1., 1.], [-4., -2.]])
    edge_index = paddle.to_tensor([[0, 1], [1, 1]])
    adj = paddle.sparse.sparse_coo_tensor(edge_index.flip([0]), edge_attr, [3, 2, 2])
    expected = [[7.5], [10.], [15.]]
    np.testing.assert_array_equal(conv((x_src, x_dst), edge_index, edge_attr).numpy(), expected)
    np.testing.assert_array_equal(conv((x_src, x_dst), adj).numpy(), expected)
    assert GINEConv(nn.Identity(), eps=0).eps.dtype == paddle.float32
    assert conv.state_dict()['eps'].stop_gradient
    assert not GINEConv(nn.Identity(), train_eps=True).eps.stop_gradient


def test_cfconv_applies_smooth_cutoff_and_reduces_along_the_node_axis():
    from models.encoder.schnet import CFConv

    conv = CFConv(1, 1, 1, nn.Identity(), cutoff=2.0, smooth=True)
    conv.lin1.weight.set_value(np.ones((1, 1), dtype=np.float32))
    conv.lin2.weight.set_value(np.ones((1, 1), dtype=np.float32))
    x = paddle.to_tensor([[[2.], [5.], [6.]], [[-2.], [5.], [6.]]])
    index = paddle.to_tensor([[0, 2], [1, 1]])
    length = paddle.to_tensor([[0.], [3.]])
    edge_attr = paddle.to_tensor([[3.], [10.]])
    expected = [[[0.], [6.], [0.]], [[0.], [-6.], [0.]]]
    csr = paddle.sparse.sparse_csr_tensor([0, 0, 2, 2], [0, 2], [1., 1.], [3, 3])
    np.testing.assert_array_equal(conv(x, index, length, edge_attr).numpy(), expected)
    np.testing.assert_array_equal(conv(x, csr, length, edge_attr).numpy(), expected)


class AliasedLayers(nn.Layer):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 1)
        self.alias = self.lin


def test_native_npz_loads_without_transposing(tmp_path):
    from models.weights import load_released_weights

    model = AliasedLayers()
    weight = np.array([[2.], [3.]], dtype=np.float32)
    bias = np.array([1.], dtype=np.float32)
    path = tmp_path / 'weights.npz'
    np.savez(path, **{'lin.weight': weight, 'lin.bias': bias, 'alias.weight': weight, 'alias.bias': bias})
    load_released_weights(model, path)
    np.testing.assert_array_equal(model.lin(paddle.to_tensor([[5., 7.]])).numpy(), [[32.]])


@pytest.mark.parametrize('defect', ['keys', 'shape', 'dtype', 'alias'])
def test_strict_state_rejects_bad_entries_before_updating_model(defect):
    from models.weights import set_model_state

    model = AliasedLayers()
    snapshot = {key: tensor.numpy().copy() for key, tensor in model.state_dict().items()}
    weight = paddle.to_tensor([[2.], [3.]])
    bias = paddle.to_tensor([1.])
    state = {'lin.weight': weight, 'lin.bias': bias, 'alias.weight': weight.clone(), 'alias.bias': bias.clone()}
    if defect == 'keys':
        state['unexpected'] = state.pop('alias.bias')
    elif defect == 'shape':
        state['alias.bias'] = bias.reshape([1, 1])
    elif defect == 'dtype':
        state['alias.bias'] = bias.astype('float64')
    else:
        state['alias.bias'] = bias + 1
    with pytest.raises(ValueError, match=defect):
        set_model_state(model, state)
    for key, actual in model.state_dict().items():
        np.testing.assert_array_equal(actual.numpy(), snapshot[key])
    assert model.lin is model.alias
