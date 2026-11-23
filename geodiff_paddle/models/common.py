import paddle
from paddle import nn
from paddle.nn import functional as F

from graph_ops import coalesce, radius, radius_graph, scatter_add, scatter_max, scatter_mean, to_dense_adj
from utils.chem import BOND_TYPES


class MeanReadout(nn.Layer):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data: batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.
            num_graphs)
        return output


class SumReadout(nn.Layer):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data: batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs
            )
        return output


class MultiLayerPerceptron(nn.Layer):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation='relu', dropout=0):
        super(MultiLayerPerceptron, self).__init__()
        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = paddle.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.layers = nn.LayerList(nn.Linear(a, b) for a, b in zip(self.dims, self.dims[1:]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row = paddle.gather(node_attr, edge_index[0])
    h_col = paddle.gather(node_attr, edge_index[1])
    h_pair = paddle.concat([h_row * h_col, edge_attr], axis=-1)
    return h_pair


def generate_symmetric_edge_noise(num_nodes_per_graph, edge_index,
    edge2graph, device):
    num_cum_nodes = num_nodes_per_graph.cumsum(0)
    node_offset = num_cum_nodes - num_nodes_per_graph
    edge_offset = node_offset[edge2graph]
    num_nodes_square = num_nodes_per_graph ** 2
    num_nodes_square_cumsum = num_nodes_square.cumsum(-1)
    edge_start = num_nodes_square_cumsum - num_nodes_square
    edge_start = edge_start[edge2graph]
    all_len = num_nodes_square_cumsum[-1]
    node_index = edge_index.transpose([1, 0]) - edge_offset.unsqueeze(-1)
    node_large = node_index.max(axis=-1)
    node_small = node_index.min(axis=-1)
    undirected_edge_id = node_large * (node_large + 1
        ) + node_small + edge_start
    symm_noise = paddle.randn([all_len.item()], device=device)
    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)
    return d_noise


def higher_order_edges(num_nodes, edge_index, edge_type, order=3):
    """Return row-major edge indices, types and shortest bond hops as int64."""
    edge_index = edge_index.astype(paddle.int64)
    edge_type = edge_type.astype(paddle.int64)
    if order <= 1 or edge_index.shape[1] == 0:
        index, types = coalesce(edge_index, edge_type, num_nodes, num_nodes)
        return index, types, paddle.ones_like(types)

    bonds = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0) > 0
    diagonal = paddle.eye(num_nodes, dtype=paddle.float32, device=edge_index.place) != 0
    adjacency = (bonds | diagonal).astype(paddle.float32)
    hop = bonds.astype(paddle.int64)
    reached = bonds | diagonal
    for distance in range(2, order + 1):
        next_reached = (reached.astype(paddle.float32) @ adjacency) > 0
        hop = paddle.where(next_reached & ~reached, distance, hop)
        reached = next_reached

    coords = paddle.nonzero(hop)
    edge_hop = paddle.gather_nd(hop, coords)
    type_mat = to_dense_adj(edge_index, edge_attr=edge_type, max_num_nodes=num_nodes).squeeze(0)
    direct_type = paddle.gather_nd(type_mat, coords)
    new_type = paddle.where(edge_hop == 1, direct_type, len(BOND_TYPES) + edge_hop - 1)
    return coords.transpose([1, 0]), new_type, edge_hop


def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch,
    unspecified_type_number=0, is_sidechain=None):
    assert edge_type.ndim == 1
    N = pos.shape[0]
    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)
    else:
        is_sidechain = is_sidechain.astype(paddle.bool)
        dummy_index = paddle.arange(N, dtype=paddle.int64, device=pos.place)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]
        assign_index = radius(x=pos, y=sidechain_pos, r=cutoff, batch_x=
            batch, batch_y=sidechain_batch)
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]
        rgraph_edge_index1 = paddle.stack((r_edge_index_x, r_edge_index_y))
        rgraph_edge_index2 = paddle.stack((r_edge_index_y, r_edge_index_x))
        rgraph_edge_index = paddle.concat(
            (rgraph_edge_index1, rgraph_edge_index2), axis=-1)
        rgraph_edge_index = rgraph_edge_index[:, rgraph_edge_index[0] !=
            rgraph_edge_index[1]]
    radius_type = paddle.full([rgraph_edge_index.shape[1]], unspecified_type_number,
                              dtype=paddle.int64, device=edge_index.place)
    return coalesce(paddle.concat((edge_index, rgraph_edge_index), axis=1),
                    paddle.concat((edge_type.astype(paddle.int64), radius_type)), N, N)


def extend_graph_order_radius(num_nodes, pos, edge_index, edge_type, batch,
    order=3, cutoff=10.0, extend_order=True, extend_radius=True,
    is_sidechain=None):
    if extend_order:
        edge_index, edge_type, _ = higher_order_edges(num_nodes, edge_index, edge_type, order)
    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(pos=pos, edge_index
            =edge_index, edge_type=edge_type, cutoff=cutoff, batch=batch,
            is_sidechain=is_sidechain)
    return edge_index, edge_type


def coarse_grain(pos, node_attr, subgraph_index, batch):
    cluster_pos = scatter_mean(pos, index=subgraph_index, dim=0)
    cluster_attr = scatter_add(node_attr, index=subgraph_index, dim=0)
    cluster_batch, _ = scatter_max(batch, index=subgraph_index, dim=0)
    return cluster_pos, cluster_attr, cluster_batch


def batch_to_natoms(batch):
    return scatter_add(paddle.ones_like(batch), index=batch, dim=0)


def get_complete_graph(natoms):
    """
    Args:
        natoms: Number of nodes per graph, (B, 1).
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of nodes of the i-th graph.
        num_edges:  (B, ), number of edges per graph.
    """
    natoms = natoms.reshape([-1])
    natoms_sqr = (natoms ** 2).astype(paddle.int64)
    num_atom_pairs = paddle.sum(natoms_sqr)
    natoms_expand = paddle.repeat_interleave(natoms, natoms_sqr)
    index_offset = paddle.cumsum(natoms, axis=0) - natoms
    index_offset_expand = paddle.repeat_interleave(index_offset, natoms_sqr)
    index_sqr_offset = paddle.cumsum(natoms_sqr, axis=0) - natoms_sqr
    index_sqr_offset = paddle.repeat_interleave(index_sqr_offset, natoms_sqr)
    atom_count_sqr = paddle.arange(
        num_atom_pairs, dtype=paddle.int64, device=natoms.place) - index_sqr_offset
    index1 = atom_count_sqr // natoms_expand + index_offset_expand
    index2 = atom_count_sqr % natoms_expand + index_offset_expand
    edge_index = paddle.stack([index1, index2])
    mask = paddle.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]
    num_edges = natoms_sqr - natoms
    return edge_index, num_edges
