import paddle
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI
from utils.chem import BOND_TYPES
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron


class GaussianSmearingEdgeEncoder(paddle.nn.Module):

    def __init__(self, num_gaussians=64, cutoff=10.0):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.rbf = GaussianSmearing(start=0.0, stop=cutoff * 2,
            num_gaussians=num_gaussians)
        self.bond_emb = paddle.nn.Embedding(100, embedding_dim=num_gaussians)

    @property
    def out_channels(self):
        return self.num_gaussians * 2

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        edge_attr = paddle.cat([self.rbf(edge_length), self.bond_emb(
            edge_type)], dim=1)
        return edge_attr


class MLPEdgeEncoder(paddle.nn.Module):

    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = paddle.nn.Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim, self.
            hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length)
        edge_attr = self.bond_emb(edge_type)
        return d_emb * edge_attr


def get_edge_encoder(cfg):
    if cfg.edge_encoder == 'mlp':
        return MLPEdgeEncoder(cfg.hidden_dim, cfg.mlp_act)
    elif cfg.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(config.hidden_dim // 2, cutoff=
            config.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % cfg.edge_encoder
            )
