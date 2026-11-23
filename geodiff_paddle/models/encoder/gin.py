import paddle
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron


class GINEConv(MessagePassing):

    def __init__(self, nn: Callable, eps: float=0.0, train_eps: bool=False,
        activation='softplus', **kwargs):
        super(GINEConv, self).__init__(aggr='add', **kwargs)
        self.nn = paddle.nn
        self.initial_eps = eps
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if train_eps:
            self.eps = paddle.nn.Parameter(paddle.Tensor([eps]))
        else:
            self.register_buffer('eps', paddle.Tensor([eps]))

    def forward(self, x: Union[paddle.Tensor, OptPairTensor], edge_index:
        Adj, edge_attr: OptTensor=None, size: Size=None) ->paddle.Tensor:
        """"""
        if isinstance(x, paddle.Tensor):
            x: OptPairTensor = (x, x)
        if isinstance(edge_index, paddle.Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: paddle.Tensor, edge_attr: paddle.Tensor
        ) ->paddle.Tensor:
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GINEncoder(paddle.nn.Module):

    def __init__(self, hidden_dim, num_convs=3, activation='relu',
        short_cut=True, concat_hidden=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = paddle.nn.Embedding(100, hidden_dim)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        self.convs = paddle.nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(GINEConv(MultiLayerPerceptron(hidden_dim, [
                hidden_dim, hidden_dim], activation=activation), activation
                =activation))

    def forward(self, z, edge_index, edge_attr):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """
        node_attr = self.node_emb(z)
        hiddens = []
        conv_input = node_attr
        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden = hidden + conv_input
            hiddens.append(hidden)
            conv_input = hidden
        if self.concat_hidden:
            node_feature = paddle.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        return node_feature
