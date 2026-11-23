"""Read compressed GeoDiff molecule arrays without Torch or pickle."""

import numpy as np
import paddle

from .data import Batch, Data


def load_graphs(path):
    """Return graphs in archive order with molecule-local edge indices."""
    with np.load(path, allow_pickle=False) as archive:
        atom_type, pos, edge_index, edge_type = (
            archive[key] for key in ('atom_type', 'pos', 'edge_index', 'edge_type'))
        nodes, edges = archive['node_ptr'], archive['edge_ptr']
        attributes = {key: archive[key] for key in ('idx', 'totalenergy', 'boltzmannweight')
                      if key in archive}
        smiles = archive['smiles'] if 'smiles' in archive else None
        if 'rdmol_ptr' in archive:
            from rdkit import Chem

            molecule_ptr = archive['rdmol_ptr']
            molecules = archive['rdmol_data'].tobytes()
        graphs = []
        for i in range(len(nodes) - 1):
            n, e = slice(nodes[i], nodes[i + 1]), slice(edges[i], edges[i + 1])
            fields = dict(atom_type=atom_type[n], pos=pos[n], edge_index=edge_index[:, e],
                          edge_type=edge_type[e])
            fields.update((key, array[i]) for key, array in attributes.items())
            graph = Data(**{key: paddle.to_tensor(value, place='cpu')
                            for key, value in fields.items()})
            if smiles is not None:
                graph.smiles = str(smiles[i])
            if 'rdmol_ptr' in archive:
                binary = molecules[molecule_ptr[i]:molecule_ptr[i + 1]]
                if binary:
                    graph.rdmol = Chem.Mol(binary)
            graphs.append(graph)
    return graphs


class GraphDataset(paddle.io.Dataset):
    """Paddle dataset backed by portable molecules; transforms receive clones."""

    def __init__(self, path, transform=None):
        super().__init__()
        self.data = load_graphs(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        graph = self.data[index].clone()
        return self.transform(graph) if self.transform is not None else graph



def collate_graphs(graphs):
    return Batch.from_data_list(graphs)


def _collate_fields(graphs):
    # Paddle's reader only sees tensors inside standard sequences/mappings.
    return (vars(collate_graphs(graphs)),)


class GraphDataLoader(paddle.io.DataLoader):
    """Use Paddle's reader and samplers while yielding graph Batch objects."""

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=_collate_fields, **kwargs)

    def __iter__(self):
        for fields, in super().__iter__():
            batch = Batch()
            batch.__dict__.update(fields)
            yield batch
