import paddle
from math import log, pi as PI
from paddle import nn

from graph_ops import scatter_add
from ._message import sparse_coo


class GaussianSmearing(nn.Layer):

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = paddle.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.reshape([-1, 1]) - self.offset.reshape([1, -1])
        return paddle.exp(self.coeff * paddle.pow(dist, 2))


class AsymmetricSineCosineSmearing(nn.Layer):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', paddle.arange(1, num_basis_k + 1, dtype='float32'))
        self.register_buffer('freq_l', paddle.arange(1, num_basis_l + 1, dtype='float32'))

    @property
    def num_basis(self):
        return self.freq_k.shape[0] + self.freq_l.shape[0]

    def forward(self, angle):
        s = paddle.sin(angle.reshape([-1, 1]) * self.freq_k.reshape([1, -1]))
        c = paddle.cos(angle.reshape([-1, 1]) * self.freq_l.reshape([1, -1]))
        return paddle.concat([s, c], axis=-1)


class SymmetricCosineSmearing(nn.Layer):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', paddle.arange(1, num_basis + 1, dtype='float32'))

    @property
    def num_basis(self):
        return self.freq_k.shape[0]

    def forward(self, angle):
        return paddle.cos(angle.reshape([-1, 1]) * self.freq_k.reshape([1, -1]))


class ShiftedSoftplus(nn.Layer):

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = log(2.0)

    def forward(self, x):
        return paddle.nn.functional.softplus(x) - self.shift


class CFConv(nn.Layer):

    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff,
        smooth):
        super().__init__()
        self.lin1 = paddle.nn.Linear(in_channels, num_filters, bias_attr=False)
        self.lin2 = paddle.nn.Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth
        self.reset_parameters()

    def reset_parameters(self):
        xavier = nn.initializer.XavierUniform()
        xavier(self.lin1.weight)
        xavier(self.lin2.weight)
        nn.initializer.Constant(0)(self.lin2.bias)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (paddle.cos(edge_length * PI / self.cutoff) + 1.0)
            within_cutoff = (edge_length <= self.cutoff) & (edge_length >= 0.0)
            C = C * within_cutoff.astype(C.dtype)
        else:
            C = (edge_length <= self.cutoff).astype(edge_length.dtype)
        W = self.nn(edge_attr) * C.reshape([-1, 1])
        x = self.lin1(x)
        adjacency = sparse_coo(edge_index)
        if adjacency is None:
            src, dst = edge_index
            n_dst = x.shape[-2]
        else:
            dst, src = adjacency.indices()
            n_dst = adjacency.shape[0]
            if adjacency.shape[1] != x.shape[-2]:
                raise ValueError('Node features do not match the sparse adjacency')
        messages = self.message(paddle.gather(x, src, axis=-2), W)
        x = self.lin2(scatter_add(messages, dst, dim=-2, dim_size=n_dst))
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(nn.Layer):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff,
        smooth):
        super(InteractionBlock, self).__init__()
        mlp = nn.Sequential(nn.Linear(num_gaussians, num_filters),
                            ShiftedSoftplus(), nn.Linear(num_filters, num_filters))
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
            mlp, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(nn.Layer):

    def __init__(self, hidden_channels=128, num_filters=128,
        num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.embedding = paddle.nn.Embedding(100, hidden_channels, max_norm
            =10.0)
        self.interactions = nn.LayerList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                num_filters, cutoff, smooth)
            self.interactions.append(block)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if embed_node:
            assert z.ndim == 1 and z.dtype == paddle.int64
            h = self.embedding(z)
        else:
            h = z
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h
