import paddle
from paddle.nn import functional as F

from graph_ops import scatter_add
from ..common import MultiLayerPerceptron
from ._message import sparse_coo


class GINEConv(paddle.nn.Layer):

    def __init__(self, nn, eps=0.0, train_eps=False, activation='softplus',
                 flow='source_to_target', node_dim=-2):
        super().__init__()
        if flow not in ('source_to_target', 'target_to_source'):
            raise ValueError(f'Invalid message flow: {flow!r}')
        self.nn = nn
        self.flow = flow
        self.node_dim = node_dim
        self.initial_eps = eps
        self.activation = getattr(F, activation) if isinstance(activation, str) else None
        if train_eps:
            self.eps = self.create_parameter(
                [1], default_initializer=paddle.nn.initializer.Constant(eps))
        else:
            self.register_buffer('eps', paddle.full([1], eps))

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if isinstance(x, paddle.Tensor):
            x = (x, x)
        x_src, x_dst = x
        adjacency = sparse_coo(edge_index)
        if adjacency is not None:
            if self.flow != 'source_to_target':
                raise ValueError('Sparse adjacency must be transposed for source-to-target flow')
            dst, src = adjacency.indices()
            size = (adjacency.shape[1], adjacency.shape[0])
            edge_attr = adjacency.values()
        else:
            src, dst = edge_index
            if self.flow == 'target_to_source':
                src, dst = dst, src
        n_src = x_src.shape[self.node_dim]
        n_dst = x_dst.shape[self.node_dim] if x_dst is not None else None
        if size is not None:
            if size[0] not in (None, n_src) or (n_dst is not None and size[1] not in (None, n_dst)):
                raise ValueError('Node features do not match the specified graph size')
            if size[1] is not None:
                n_dst = size[1]
        if n_dst is None:
            n_dst = n_src
        assert edge_attr is not None and x_src.shape[-1] == edge_attr.shape[-1]
        messages = self.message(paddle.gather(x_src, src, axis=self.node_dim), edge_attr)
        out = scatter_add(messages, dst, dim=self.node_dim, dim_size=n_dst)
        if x_dst is not None:
            out = out + (1 + self.eps) * x_dst
        return self.nn(out)

    def message(self, x_j: paddle.Tensor, edge_attr: paddle.Tensor
        ) ->paddle.Tensor:
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GINEncoder(paddle.nn.Layer):

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
        self.convs = paddle.nn.LayerList()
        for i in range(self.num_convs):
            self.convs.append(GINEConv(MultiLayerPerceptron(hidden_dim, [
                hidden_dim, hidden_dim], activation=activation), activation
                =activation))

    def forward(self, z, edge_index, edge_attr):
        """
        Input:
            z: atom indices for the batched graph
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
            node_feature = paddle.concat(hiddens, axis=-1)
        else:
            node_feature = hiddens[-1]
        return node_feature
