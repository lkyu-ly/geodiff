"""Torch-free graph containers and portable molecular assets."""

import networkx as nx
import paddle

from .assets import GraphDataLoader, GraphDataset, collate_graphs, load_graphs
from .data import Batch, Data

__all__ = ['Data', 'Batch', 'Compose', 'to_networkx', 'GraphDataset', 'GraphDataLoader',
           'collate_graphs', 'load_graphs']


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        args = '\n'.join(f'    {transform},' for transform in self.transforms)
        return f'{type(self).__name__}([\n{args}\n])'


def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    """Convert a graph, retaining isolated nodes and PyG's triangle selection."""
    graph = nx.Graph() if to_undirected else nx.DiGraph()

    def attributes(keys):
        return {key: data[key].tolist() if isinstance(data[key], paddle.Tensor) else data[key]
                for key in keys or ()}

    nodes, edges = attributes(node_attrs), attributes(edge_attrs)
    graph.add_nodes_from((i, {key: values[i] for key, values in nodes.items()})
                         for i in range(data.num_nodes))
    for i, (source, target) in enumerate(data.edge_index.transpose([1, 0]).tolist()):
        if (to_undirected and target > source) or (remove_self_loops and source == target):
            continue
        graph.add_edge(source, target, **{key: values[i] for key, values in edges.items()})
    return graph
