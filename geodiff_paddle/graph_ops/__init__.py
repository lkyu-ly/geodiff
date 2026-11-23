"""Paddle graph operators used by GeoDiff."""

from .adjacency import coalesce, dense_to_sparse, to_dense_adj
from .radius import radius, radius_graph
from .reduction import scatter, scatter_add, scatter_max, scatter_mean

__all__ = [
    "coalesce",
    "dense_to_sparse",
    "radius",
    "radius_graph",
    "scatter",
    "scatter_add",
    "scatter_max",
    "scatter_mean",
    "to_dense_adj",
]
