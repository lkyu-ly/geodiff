import copy

import paddle

from graph_data import Data
from graph_ops import to_dense_adj
from models.common import higher_order_edges

from .chem import BOND_NAMES, BOND_TYPES, get_atom_symbol


class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=len(BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def __call__(self, data: Data):
        data.bond_edge_index = data.edge_index
        data.edge_index, data.edge_type, data.edge_order = higher_order_edges(
            data.num_nodes, data.edge_index, data.edge_type, self.order
        )
        data.is_bond = data.edge_order == 1
        if self.num_types != len(BOND_TYPES):
            data.edge_type = paddle.where(data.is_bond, data.edge_type,
                                          self.num_types + data.edge_order - 1)
        return data


class AddEdgeLength(object):

    def __call__(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = paddle.linalg.norm(pos[row] - pos[col], axis=-1).unsqueeze(-1)
        data.edge_length = d
        return data


class AddPlaceHolder(object):

    def __call__(self, data: Data):
        data.pos_gen = -1.0 * paddle.ones_like(data.pos)
        data.d_gen = -1.0 * paddle.ones_like(data.edge_length)
        data.d_recover = -1.0 * paddle.ones_like(data.edge_length)
        return data


class AddEdgeName(object):

    def __init__(self, asymmetric=True):
        super().__init__()
        self.bonds = copy.deepcopy(BOND_NAMES)
        self.bonds[len(BOND_NAMES) + 1] = 'Angle'
        self.bonds[len(BOND_NAMES) + 2] = 'Dihedral'
        self.asymmetric = asymmetric

    def __call__(self, data: Data):
        data.edge_name = []
        for i in range(data.edge_index.shape[1]):
            tail = data.edge_index[0, i].item()
            head = data.edge_index[1, i].item()
            if self.asymmetric and tail >= head:
                data.edge_name.append('')
                continue
            tail_name = get_atom_symbol(data.atom_type[tail].item())
            head_name = get_atom_symbol(data.atom_type[head].item())
            name = '%s_%s_%s_%d_%d' % (self.bonds[data.edge_type[i].item()] if
                data.edge_type[i].item() in self.bonds else 'E' + str(data.
                edge_type[i].item()), tail_name, head_name, tail, head)
            if 'edge_length' in data:
                name += '_%.3f' % data.edge_length[i].item()
            data.edge_name.append(name)
        return data


class AddAngleDihedral(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def iter_angle_triplet(bond_mat):
        n_atoms = bond_mat.shape[0]
        for j in range(n_atoms):
            for k in range(n_atoms):
                for l in range(n_atoms):
                    if bond_mat[j, k].item() == 0 or bond_mat[k, l].item(
                        ) == 0:
                        continue
                    if j == k or k == l or j >= l:
                        continue
                    yield j, k, l

    @staticmethod
    def iter_dihedral_quartet(bond_mat):
        n_atoms = bond_mat.shape[0]
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i >= j:
                    continue
                if bond_mat[i, j].item() == 0:
                    continue
                for k in range(n_atoms):
                    for l in range(n_atoms):
                        if k in (i, j) or l in (i, j):
                            continue
                        if bond_mat[k, i].item() == 0 or bond_mat[l, j].item(
                            ) == 0:
                            continue
                        yield k, i, j, l

    def __call__(self, data: Data):
        edge_attr = data.is_bond if 'is_bond' in data else data.edge_type
        bond_mat = to_dense_adj(data.edge_index, edge_attr=edge_attr,
                                max_num_nodes=data.num_nodes).squeeze(0) > 0
        data.angle_index = paddle.to_tensor(list(self.iter_angle_triplet(bond_mat)),
                                            dtype='int64', place=bond_mat.place).reshape([-1, 3]).T
        data.dihedral_index = paddle.to_tensor(list(self.iter_dihedral_quartet(bond_mat)),
                                               dtype='int64', place=bond_mat.place).reshape([-1, 4]).T
        return data


class CountNodesPerGraph(object):

    def __init__(self) ->None:
        super().__init__()

    def __call__(self, data):
        place = data.pos.place if 'pos' in data else None
        data.num_nodes_per_graph = paddle.to_tensor([data.num_nodes], dtype='int64', place=place)
        return data
