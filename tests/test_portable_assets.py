import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class PortableAssetsTest(unittest.TestCase):
    def test_numeric_npz_batches_and_separates_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'graphs.npz'
            np.savez(
                target,
                atom_type=np.array([8, 6, 1, 7, 9], dtype=np.int32),
                pos=np.r_[0:9, 0:6].astype(np.float64).reshape(-1, 3),
                edge_index=np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int64),
                edge_type=np.array([2, 2, 1], dtype=np.int16),
                node_ptr=np.array([0, 3, 5], dtype=np.int64),
                edge_ptr=np.array([0, 2, 3], dtype=np.int64),
            )
            self.assert_portable_batch(target)

    @unittest.skipUnless(os.environ.get('GEODIFF_TORCH_PYTHON'),
                         'Set GEODIFF_TORCH_PYTHON to the separate Torch interpreter')
    def test_torch_export_preserves_portable_graphs_and_chemistry(self):
        from rdkit import Chem

        with tempfile.TemporaryDirectory() as tmp:
            source, target = (Path(tmp) / name for name in ('graphs.pkl', 'graphs.npz'))
            torch_python = os.environ['GEODIFF_TORCH_PYTHON']
            subprocess.run([torch_python, '-c', '''
import pickle, sys, torch
from torch_geometric.data import Data
from rdkit import Chem

params = Chem.SmilesParserParams()
params.removeHs = False

def graph(atoms, edges, bonds, smiles):
    return Data(atom_type=torch.tensor(atoms, dtype=torch.int32),
                pos=torch.arange(len(atoms)*3, dtype=torch.float64).reshape(-1, 3),
                edge_index=torch.tensor(edges, dtype=torch.long).reshape(2, -1),
                edge_type=torch.tensor(bonds, dtype=torch.int16), smiles=smiles,
                rdmol=Chem.MolFromSmiles(smiles, params))
with open(sys.argv[1], 'wb') as f:
    pickle.dump([graph([8, 6, 1], [[1, 0], [0, 1]], [2, 2], 'OC[H]'),
                 graph([7, 9], [[0], [1]], [1], 'NF'),
                 graph([6], [[], []], [], 'C')], f)
''', str(source)], check=True)
            subprocess.run([torch_python, str(ROOT / 'tools/export_paddle_assets.py'),
                            'graphs', str(source), str(target), '--limit', '2'], check=True)
            graphs = self.assert_portable_batch(target)
            self.assertEqual([graph.smiles for graph in graphs], ['OC[H]', 'NF'])
            self.assertEqual([Chem.MolToSmiles(Chem.RemoveHs(graph.rdmol)) for graph in graphs],
                             ['CO', 'NF'])
            self.assertEqual([[atom.GetAtomicNum() for atom in graph.rdmol.GetAtoms()]
                              for graph in graphs], [[8, 6, 1], [7, 9]])

    def assert_portable_batch(self, target):
        from geodiff_paddle.graph_data import GraphDataLoader, GraphDataset

        batch, = GraphDataLoader(GraphDataset(target), batch_size=2)
        self.assertEqual((batch.num_nodes, batch.num_edges, batch.num_graphs), (5, 3, 2))
        np.testing.assert_array_equal(batch.batch.numpy(), [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(batch.edge_index.numpy(), [[1, 0, 3], [0, 1, 4]])
        np.testing.assert_array_equal(batch.atom_type.numpy(), [8, 6, 1, 7, 9])
        np.testing.assert_array_equal(batch.edge_type.numpy(), [2, 2, 1])
        np.testing.assert_array_equal(batch.pos.numpy(), np.r_[0:9, 0:6].reshape(-1, 3))
        self.assertEqual([str(getattr(batch, k).numpy().dtype) for k in
                          ('atom_type', 'edge_index', 'edge_type', 'pos')],
                         ['int32', 'int64', 'int16', 'float64'])
        first, second = batch.clone().cpu().to_data_list()
        np.testing.assert_array_equal(first.edge_index.numpy(), [[1, 0], [0, 1]])
        np.testing.assert_array_equal(second.edge_index.numpy(), [[0], [1]])
        np.testing.assert_array_equal(second.atom_type.numpy(), [7, 9])
        self.assertNotIn('torch', sys.modules)
        return first, second


    def test_subclass_indices_and_empty_edges_survive_batching(self):
        import paddle
        from geodiff_paddle.graph_data import Compose, Data, collate_graphs, to_networkx

        class MolClusterData(Data):
            def __inc__(self, key, value):
                if key == 'subgraph_index':
                    return int(value.max().item()) + 1
                return super().__inc__(key, value)

        a = MolClusterData(atom_type=paddle.to_tensor([6, 1, 8]),
                           edge_index=paddle.empty([2, 0], dtype='int64'),
                           subgraph_index=paddle.to_tensor([0, 1, 1]))
        b = MolClusterData(atom_type=paddle.to_tensor([7, 6]),
                           edge_index=paddle.to_tensor([[1], [0]]),
                           subgraph_index=paddle.to_tensor([0, 0]), smiles='NC')
        batch = Compose([collate_graphs])([a, b])
        np.testing.assert_array_equal(batch.subgraph_index.numpy(), [0, 1, 1, 2, 2])
        np.testing.assert_array_equal(batch.edge_index.numpy(), [[4], [3]])
        first, second = batch.to_data_list()
        self.assertIsInstance(second, MolClusterData)
        self.assertEqual((first.num_nodes, first.num_edges), (3, 0))
        self.assertEqual(second.smiles, 'NC')
        self.assertNotIn('smiles', first)
        np.testing.assert_array_equal(second.subgraph_index.numpy(), [0, 0])
        graph = to_networkx(second, node_attrs=['atom_type'], to_undirected=True)
        self.assertEqual(list(graph.edges), [(0, 1)])
        self.assertEqual(graph.nodes[0]['atom_type'], 7)

if __name__ == '__main__':
    unittest.main()
