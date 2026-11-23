import paddle


def _reduced_shape(src, index, dim, dim_size):
    shape = list(src.shape)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if src.numel() else 0
    shape[dim] = dim_size
    return shape


def _empty_sum(src, shape):
    out = paddle.zeros(shape, dtype=src.dtype, device=src.place)
    return out if src.stop_gradient else out + src.sum()


def scatter_add(src, index, dim=-1, dim_size=None):
    """Sum entries into bins along an axis using a one-dimensional index."""
    shape = _reduced_shape(src, index, dim, dim_size)
    if src.numel() == 0:
        return _empty_sum(src, shape)
    boolean = src.dtype == paddle.bool
    if boolean:
        src = src.astype(paddle.int32)
    out = paddle.zeros(shape, dtype=src.dtype, device=src.place)
    out = paddle.index_add(out, index, dim, src)
    return out != 0 if boolean else out


def scatter_mean(src, index, dim=-1, dim_size=None):
    """Average bins, leaving empty bins at zero and floor-dividing integers."""
    out = scatter_add(src, index, dim, dim_size)
    if src.numel() == 0:
        return out
    count = scatter_add(paddle.ones_like(index, dtype=src.dtype), index, dim_size=out.shape[dim])
    shape = [1] * src.ndim
    shape[dim] = out.shape[dim]
    count = count.clip(min=1).reshape(shape)
    if paddle.is_floating_point(out):
        return out / count
    return paddle.floor_divide(out, count)


def scatter_max(src, index, dim=-1, dim_size=None):
    """Return each bin's maximum and source index; empty indices use the axis length."""
    shape = _reduced_shape(src, index, dim, dim_size)
    length = src.shape[dim]
    arg = paddle.full(shape, length, dtype="int64", device=src.place)
    if src.numel() == 0:
        return _empty_sum(src, shape), arg

    index_shape = [1] * src.ndim
    index_shape[dim] = length
    index = index.reshape(index_shape).expand(src.shape)
    floor = (paddle.finfo if paddle.is_floating_point(src) else paddle.iinfo)(src.dtype).min
    initial = paddle.full(shape, floor, dtype=src.dtype, device=src.place)
    peak = paddle.put_along_axis(initial, index, src.detach(), dim, reduce="amax", broadcast=False)
    zero = paddle.zeros([], dtype=src.dtype, device=src.place)
    peak = paddle.where(peak == floor, zero, peak)

    # Route the gradient to one winning source, not to every tied value.
    positions = paddle.arange(length, dtype="int64", device=src.place).reshape(index_shape)
    winners = paddle.take_along_axis(peak, index, dim) == src
    limit = paddle.full([], length, dtype="int64", device=src.place)
    arg = paddle.put_along_axis(
        arg, index, paddle.where(winners, positions, limit), dim, reduce="amin", broadcast=False
    )
    if not src.stop_gradient:
        found = arg != length
        safe_arg = paddle.where(found, arg, paddle.zeros_like(limit))
        peak = paddle.where(found, paddle.take_along_axis(src, safe_arg, dim), zero)
    return peak, arg


def scatter(src, index, dim=-1, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, dim_size)
    if reduce == "max":
        return scatter_max(src, index, dim, dim_size)[0]
    raise ValueError(f"Unsupported scatter reduction: {reduce}")
