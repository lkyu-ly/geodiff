import paddle

from .reduction import _empty_sum, scatter_add


def coalesce(index, value, m, n):
    """Sort the COO coordinates of an m-by-n matrix and sum duplicates."""
    if index.shape[1] == 0:
        return index, value
    keys = index[0] * n + index[1]
    keys, groups = paddle.unique(keys, return_inverse=True, sorted=True)
    index = paddle.stack((keys // n, keys % n), axis=0)
    if value is not None:
        value = scatter_add(value, groups, dim=0, dim_size=keys.shape[0])
    return index, value


def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    """Accumulate COO edges into dense adjacency, respecting explicit node size."""
    row, col = edge_index[0], edge_index[1]
    if edge_attr is None:
        edge_attr = paddle.ones([row.shape[0]], device=edge_index.place)

    if batch is None:
        size = max_num_nodes
        if size is None:
            size = int(edge_index.max()) + 1 if row.shape[0] else 0
        n_graphs = 1
    elif batch.shape[0] == 0:
        n_graphs = 0
        size = max_num_nodes if max_num_nodes is not None else 0
    else:
        counts = paddle.bincount(batch)
        n_graphs = counts.shape[0]
        size = int(counts.max()) if max_num_nodes is None else max_num_nodes

    shape = [n_graphs, size, size, *edge_attr.shape[1:]]
    if edge_attr.numel() == 0:
        return _empty_sum(edge_attr, shape)

    graph = 0
    if batch is not None:
        offsets = counts.cumsum(axis=0) - counts
        graph = paddle.gather(batch, row)
        row = row - paddle.gather(offsets, graph)
        col = col - paddle.gather(offsets, paddle.gather(batch, col))

    flat = (graph * size + row) * size + col
    adj = scatter_add(edge_attr, flat, dim=0, dim_size=n_graphs * size * size)
    return adj.reshape(shape)


def dense_to_sparse(adj):
    """Return the nonzero entries of a square matrix or batch of matrices."""
    assert adj.ndim in (2, 3) and adj.shape[-1] == adj.shape[-2]
    index = paddle.nonzero(adj)
    edge_attr = paddle.gather_nd(adj, index)
    if adj.ndim == 3:
        offset = index[:, :1] * adj.shape[-1]
        index = index[:, 1:] + offset
    return index.transpose([1, 0]), edge_attr
