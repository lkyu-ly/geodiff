import paddle
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI
from utils.chem import BOND_TYPES
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron


class GaussianSmearing(paddle.nn.Module):

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = paddle.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return paddle.exp(self.coeff * paddle.pow(dist, 2))


class AsymmetricSineCosineSmearing(paddle.nn.Module):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', paddle.arange(1, num_basis_k + 1).
            float())
        self.register_buffer('freq_l', paddle.arange(1, num_basis_l + 1).
            float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        s = paddle.sin(angle.view(-1, 1) * self.freq_k.view(1, -1))
        c = paddle.cos(angle.view(-1, 1) * self.freq_l.view(1, -1))
        return paddle.cat([s, c], dim=-1)


class SymmetricCosineSmearing(paddle.nn.Module):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', paddle.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return paddle.cos(angle.view(-1, 1) * self.freq_k.view(1, -1))


class ShiftedSoftplus(paddle.nn.Module):

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = paddle.log(paddle.tensor(2.0)).item()

    def forward(self, x):
        return paddle.nn.functional.softplus(x) - self.shift


class CFConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff,
        smooth):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = paddle.compat.nn.Linear(in_channels, num_filters, bias=
            False)
        self.lin2 = paddle.compat.nn.Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth
        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.init.xavier_uniform_(self.lin1.weight)
        paddle.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (paddle.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(paddle.nn.Module):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff,
        smooth):
        super(InteractionBlock, self).__init__()
        mlp = paddle.nn.Sequential(paddle.compat.nn.Linear(num_gaussians,
            num_filters), ShiftedSoftplus(), paddle.compat.nn.Linear(
            num_filters, num_filters))
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
            mlp, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = paddle.compat.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(paddle.nn.Module):

    def __init__(self, hidden_channels=128, num_filters=128,
        num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.embedding = paddle.nn.Embedding(100, hidden_channels, max_norm
            =10.0)
        self.interactions = paddle.nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                num_filters, cutoff, smooth)
            self.interactions.append(block)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if embed_node:
            assert z.dim() == 1 and z.dtype == paddle.long
            h = self.embedding(z)
        else:
            h = z
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h
