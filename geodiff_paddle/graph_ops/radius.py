import paddle


@paddle.no_grad()
def _radius(x, y, r, batch_x, batch_y, max_num_neighbors, ignore_same_index):
    if x.numel() == 0 or y.numel() == 0 or max_num_neighbors == 0:
        return paddle.empty([2, 0], dtype="int64", device=x.place)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if batch_x is None and batch_y is None:
        sources = paddle.arange(x.shape[0], dtype="int64", device=x.place)
        points = x.unsqueeze(0)
    else:
        # Each query compares only against the sorted source interval for its batch.
        start = paddle.searchsorted(batch_x, batch_y).unsqueeze(1)
        stop = paddle.searchsorted(batch_x, batch_y, right=True).unsqueeze(1)
        width = int((stop - start).max())
        if width == 0:
            return paddle.empty([2, 0], dtype="int64", device=x.place)
        sources = start + paddle.arange(width, dtype="int64", device=x.place)
        points = paddle.gather(x, sources.clip(max=x.shape[0] - 1).flatten()).reshape(
            [y.shape[0], width, x.shape[1]]
        )

    valid = (points - y.unsqueeze(1)).square().sum(axis=-1) < r * r
    if batch_x is not None:
        valid = valid & (sources < stop)
    if ignore_same_index:
        queries = paddle.arange(y.shape[0], dtype="int64", device=y.place).unsqueeze(1)
        valid = valid & (sources != queries)
    rank = valid.astype(paddle.int32).cumsum(axis=1, dtype="int32")
    pairs = paddle.nonzero(valid & (rank <= max_num_neighbors))
    query, source = pairs[:, 0], pairs[:, 1]
    if batch_x is not None:
        source = source + paddle.gather(start.squeeze(1), query)
    return paddle.stack((query, source), axis=0)


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    """Return (query in y, source in x), preserving the first matching source indices."""
    return _radius(x, y, r, batch_x, batch_y, max_num_neighbors, False)


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target"):
    """Return radius edges, excluding same-index pairs before capping when loop is false."""
    assert flow in ("source_to_target", "target_to_source")
    edges = _radius(x, x, r, batch, batch, max_num_neighbors, not loop)
    if flow == "source_to_target":
        return paddle.stack((edges[1], edges[0]), axis=0)
    return edges
