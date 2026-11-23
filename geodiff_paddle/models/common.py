import paddle
from torch_geometric.nn import radius_graph, radius
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils.chem import BOND_TYPES


class MeanReadout(paddle.nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.
            num_graphs)
        return output


class SumReadout(paddle.nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs
            )
        return output


class MultiLayerPerceptron(paddle.nn.Module):
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
        self.layers = paddle.nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(paddle.compat.nn.Linear(self.dims[i], self.
                dims[i + 1]))

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
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = paddle.cat([h_row * h_col, edge_attr], dim=-1)
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
    node_index = edge_index.t() - edge_offset.unsqueeze(-1)
    node_large = (node_index.max(axis=-1), node_index.argmax(axis=-1))[0]
    node_small = (node_index.min(axis=-1), node_index.argmin(axis=-1))[0]
    undirected_edge_id = node_large * (node_large + 1
        ) + node_small + edge_start
    symm_noise = paddle.zeros(size=[all_len.item()], device=device)
    symm_noise.normal_()
    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)
    return d_noise


def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """

    def binarize(x):
        return paddle.where(x > 0, paddle.ones_like(x), paddle.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [paddle.eye(adj.size(0), dtype=paddle.long, device=adj.
            device), binarize(adj + paddle.eye(adj.size(0), dtype=paddle.
            long, device=adj.device))]
        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = paddle.zeros_like(adj)
        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
        return order_mat
    num_types = len(BOND_TYPES)
    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)
    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)
    type_highorder = paddle.where(adj_order > 1, num_types + adj_order - 1,
        paddle.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder
    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.
        long(), N, N)
    return new_edge_index, new_edge_type


def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch,
    unspecified_type_number=0, is_sidechain=None):
    assert edge_type.dim() == 1
    N = pos.size(0)
>>>>>>    bgraph_adj = torch.sparse.LongTensor(edge_index, edge_type, paddle.Size
        ([N, N]))
    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)
    else:
        is_sidechain = is_sidechain.bool()
        dummy_index = paddle.arange(pos.size(0), device=pos.device)
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
        rgraph_edge_index = paddle.cat((rgraph_edge_index1,
            rgraph_edge_index2), dim=-1)
        rgraph_edge_index = rgraph_edge_index[:, rgraph_edge_index[0] !=
            rgraph_edge_index[1]]
>>>>>>    rgraph_adj = torch.sparse.LongTensor(rgraph_edge_index, paddle.ones(
        rgraph_edge_index.size(1)).long().to(pos.device) *
        unspecified_type_number, paddle.Size([N, N]))
    composed_adj = (bgraph_adj + rgraph_adj).coalesce()
    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()
    return new_edge_index, new_edge_type


def extend_graph_order_radius(num_nodes, pos, edge_index, edge_type, batch,
    order=3, cutoff=10.0, extend_order=True, extend_radius=True,
    is_sidechain=None):
    if extend_order:
        edge_index, edge_type = _extend_graph_order(num_nodes=num_nodes,
            edge_index=edge_index, edge_type=edge_type, order=order)
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
    natoms_sqr = (natoms ** 2).long()
    num_atom_pairs = paddle.sum(natoms_sqr)
    natoms_expand = paddle.repeat_interleave(natoms, natoms_sqr)
    index_offset = paddle.cumsum(natoms, dim=0) - natoms
    index_offset_expand = paddle.repeat_interleave(index_offset, natoms_sqr)
    index_sqr_offset = paddle.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = paddle.repeat_interleave(index_sqr_offset, natoms_sqr)
    atom_count_sqr = paddle.arange(num_atom_pairs, device=num_atom_pairs.device
        ) - index_sqr_offset
    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = paddle.cat([index1.view(1, -1), index2.view(1, -1)])
    mask = paddle.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]
    num_edges = natoms_sqr - natoms
    return edge_index, num_edges
