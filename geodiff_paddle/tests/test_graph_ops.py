from importlib import import_module

import numpy as np
import paddle
import pytest


@pytest.fixture
def ops():
    return import_module("geodiff_paddle.graph_ops")


def test_scatter_reductions_keep_missing_bins_and_gradients(ops):
    index = paddle.to_tensor([2, 0, 2, 0])
    src = paddle.to_tensor([[-2.0, 5.0], [-3.0, -1.0], [-4.0, 1.0], [-2.0, 7.0]], stop_gradient=False)
    total = ops.scatter_add(src, index, dim=0, dim_size=4)
    mean = ops.scatter_mean(src, index, dim=0, dim_size=4)
    peak, arg = ops.scatter_max(src, index, dim=0, dim_size=4)
    np.testing.assert_array_equal(total.numpy(), [[-5, 6], [0, 0], [-6, 6], [0, 0]])
    np.testing.assert_array_equal(mean.numpy(), [[-2.5, 3], [0, 0], [-3, 3], [0, 0]])
    np.testing.assert_array_equal(peak.numpy(), [[-2, 7], [0, 0], [-2, 5], [0, 0]])
    np.testing.assert_array_equal(arg.numpy(), [[3, 3], [4, 4], [0, 0], [4, 4]])
    (total + mean + peak).sum().backward()
    np.testing.assert_array_equal(src.grad.numpy(), [[2.5, 2.5], [1.5, 1.5], [1.5, 1.5], [2.5, 2.5]])

    values = paddle.to_tensor([[-3, -2, 5, 1], [1, 3, 4, 0]])
    bins = paddle.to_tensor([0, 0, 1, 1])
    np.testing.assert_array_equal(ops.scatter(values, bins, dim=-1, reduce="mean").numpy(), [[-3, 3], [2, 2]])
    flags = paddle.to_tensor([True, True, False, True])
    np.testing.assert_array_equal(ops.scatter(flags, bins, dim_size=3).numpy(), [True, True, False])


def test_coalesce_sorts_coordinates_and_keeps_zero_types(ops):
    index = paddle.to_tensor([[2, 0, 2, 1, 0], [0, 3, 0, 1, 3]])
    value = paddle.to_tensor([4.0, 0.0, -4.0, 0.0, 3.0], stop_gradient=False)
    merged, types = ops.coalesce(index, value, 4, 5)
    np.testing.assert_array_equal(merged.numpy(), [[0, 1, 2], [3, 1, 0]])
    np.testing.assert_array_equal(types.numpy(), [3, 0, 0])
    (types * paddle.to_tensor([1.0, 2.0, 3.0])).sum().backward()
    np.testing.assert_array_equal(value.grad.numpy(), [3, 1, 3, 2, 1])


def test_adjacency_conversions_sum_duplicates_and_keep_explicit_size(ops):
    edge = paddle.to_tensor([[1, 0, 0], [0, 1, 1]])
    attr = paddle.to_tensor([3.0, 2.0, -1.0], stop_gradient=False)
    dense = ops.to_dense_adj(edge, edge_attr=attr, max_num_nodes=3)
    np.testing.assert_array_equal(dense.numpy(), [[[0, 1, 0], [3, 0, 0], [0, 0, 0]]])
    coords, values = ops.dense_to_sparse(dense)
    np.testing.assert_array_equal(coords.numpy(), [[0, 1], [1, 0]])
    np.testing.assert_array_equal(values.numpy(), [1, 3])
    (values * paddle.to_tensor([2.0, 5.0])).sum().backward()
    np.testing.assert_array_equal(attr.grad.numpy(), [5, 2, 2])

    batched = ops.to_dense_adj(paddle.to_tensor([[0, 3], [1, 2]]), paddle.to_tensor([0, 0, 1, 1, 1]))
    assert list(batched.shape) == [2, 3, 3]
    np.testing.assert_array_equal(ops.dense_to_sparse(batched)[0].numpy(), [[0, 4], [1, 3]])


def test_empty_edges_keep_shapes_and_dtypes(ops):
    index = paddle.empty([2, 0], dtype="int64")
    vals = paddle.empty([0, 2], dtype="float32", requires_grad=True)
    sums = ops.scatter_add(vals, index[0], dim=0, dim_size=3)
    np.testing.assert_array_equal(sums.numpy(), np.zeros([3, 2]))
    sums.sum().backward()
    assert list(vals.grad.shape) == [0, 2]
    assert [list(item.shape) for item in ops.coalesce(index, vals, 3, 3)] == [[2, 0], [0, 2]]
    assert list(ops.to_dense_adj(index).shape) == [1, 0, 0]
    dense = ops.to_dense_adj(index, edge_attr=vals, max_num_nodes=3)
    assert list(dense.shape) == [1, 3, 3, 2]
    coords, values = ops.dense_to_sparse(dense[..., 0])
    assert list(coords.shape) == [2, 0] and coords.dtype == paddle.int64
    assert list(values.shape) == [0] and values.dtype == paddle.float32


def test_radius_is_strict_and_isolates_batches(ops):
    pos = paddle.to_tensor([0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 4.0])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 3])
    graph = ops.radius_graph(pos, r=1.0, batch=batch)
    np.testing.assert_array_equal(graph.numpy(), [[1, 2, 0, 2, 0, 1, 3, 2, 5, 4], [0, 0, 1, 1, 2, 2, 2, 3, 4, 5]])
    sidechain = paddle.to_tensor([1, 4])
    raw = ops.radius(x=pos, y=pos[sidechain], r=1.0, batch_x=batch, batch_y=batch[sidechain])
    np.testing.assert_array_equal(raw.numpy(), [[0, 0, 0, 1, 1], [0, 1, 2, 4, 5]])
    assert list(ops.radius_graph(pos[:0], r=1.0).shape) == [2, 0]
    assert list(ops.radius(pos, pos[:0], r=1.0).shape) == [2, 0]


def test_radius_caps_in_source_order_after_excluding_self(ops):
    pos = paddle.arange(35, dtype="float32") / 100
    graph = ops.radius_graph(pos, r=1.0).numpy()
    assert graph.shape == (2, 1120)
    np.testing.assert_array_equal(np.bincount(graph[1]), [32] * 35)
    np.testing.assert_array_equal(graph[0, graph[1] == 0], np.arange(1, 33))
    np.testing.assert_array_equal(graph[0, graph[1] == 34], np.arange(32))
    raw = ops.radius(x=pos, y=pos[paddle.to_tensor([0, 34])], r=1.0).numpy()
    assert raw.shape == (2, 64)
    np.testing.assert_array_equal(raw[0], [0] * 32 + [1] * 32)
    np.testing.assert_array_equal(raw[1], list(range(32)) * 2)
