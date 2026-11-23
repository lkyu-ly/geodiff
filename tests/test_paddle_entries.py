import pickle

import numpy as np
import paddle
import pytest
from rdkit import Chem


def test_covmat_uses_both_axes_of_rectangular_rmsd_matrix():
    from graph_data import Data
    from utils.evaluation.covmat import CovMatEvaluator, evaluate_conf

    mol = Chem.MolFromSmiles('CO')
    mol.AddConformer(Chem.Conformer(2))
    data = Data(rdmol=mol, smiles='CO',
                pos_ref=np.array([[[0., 0., 0.], [1., 0., 0.]],
                                  [[0., 0., 0.], [3., 0., 0.]]]),
                pos_gen=np.array([[[0., 0., 0.], [1., 0., 0.]],
                                  [[0., 0., 0.], [2., 0., 0.]],
                                  [[0., 0., 0.], [5., 0., 0.]],
                                  [[0., 0., 0.], [7., 0., 0.]]]))
    assert evaluate_conf(data, threshold=0.25) == (0.5, 0.25)
    evaluator = CovMatEvaluator(num_workers=1, thresholds=[0.25, 1.25])
    with evaluator.pool:
        results = evaluator([data])
    np.testing.assert_allclose(results.CoverageR, [[0.5, 1.]])
    np.testing.assert_allclose(results.CoverageP, [[0.25, 0.75]])
    np.testing.assert_allclose(results.MatchingR, [0.25])
    np.testing.assert_allclose(results.MatchingP, [0.875])


def test_packed_npz_retains_smiles_groups_conformer_order_and_repeats(tmp_path):
    from utils.datasets import PackedConformationDataset
    from utils.misc import repeat_batch, repeat_data

    mol = Chem.MolFromSmiles('OC')
    mol.AddConformer(Chem.Conformer(2))
    first = mol.ToBinary()
    last = Chem.MolFromSmiles('N').ToBinary()
    path = tmp_path / 'test.npz'
    np.savez(path, atom_type=[8, 6, 7, 8, 6],
             pos=np.array([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.],
                           [4., 0., 0.], [5., 0., 0.]], dtype=np.float32),
             edge_index=[[0, 1, 0, 1], [1, 0, 1, 0]], edge_type=[1, 1, 1, 1],
             node_ptr=[0, 2, 3, 5], edge_ptr=[0, 2, 2, 4],
             smiles=['CO', 'N', 'CO'],
             rdmol_data=np.frombuffer(first + last + first, dtype=np.uint8),
             rdmol_ptr=[0, len(first), len(first) + len(last), 2 * len(first) + len(last)])
    packed = PackedConformationDataset(path)
    assert [graph.smiles for graph in packed] == ['CO', 'N']
    graph = packed[0]
    assert graph.num_pos_ref.item() == 2
    assert [a.GetAtomicNum() for a in graph.rdmol.GetAtoms()] == [8, 6]
    np.testing.assert_array_equal(graph.pos_ref.numpy()[:, 0], [1., 2., 4., 5.])
    data = repeat_batch(repeat_data(graph, 2), 2)
    np.testing.assert_array_equal(data.ptr.numpy(), [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(data.edge_index.numpy(),
                                  [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]])


def test_transform_keeps_zero_type_bond_and_adds_hops_with_isolate():
    from graph_data import Batch, Compose, Data
    from utils.chem import BOND_TYPES
    from utils.transforms import AddAngleDihedral, AddEdgeLength, AddHigherOrderEdges, CountNodesPerGraph

    graph = Data(atom_type=paddle.to_tensor([6, 6, 6, 8]),
                 edge_index=paddle.to_tensor([[2, 0, 1, 1], [1, 1, 0, 2]]),
                 edge_type=paddle.to_tensor([1, 0, 0, 1]),
                 pos=paddle.to_tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [0., 0., 0.]]))
    transform = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=2),
                         AddEdgeLength(), AddAngleDihedral()])
    output = transform(graph)
    np.testing.assert_array_equal(output.edge_index.numpy(), [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    np.testing.assert_array_equal(output.edge_type.numpy(), [0, len(BOND_TYPES) + 1, 0, 1, len(BOND_TYPES) + 1, 1])
    np.testing.assert_array_equal(output.edge_order.numpy(), [1, 2, 1, 1, 2, 1])
    np.testing.assert_array_equal(output.edge_length.numpy().ravel(), [1., 3., 1., 2., 3., 2.])
    np.testing.assert_array_equal(output.angle_index.numpy(), [[0], [1], [2]])
    assert tuple(output.dihedral_index.shape) == (4, 0)
    assert Batch.from_data_list([output]).num_nodes_per_graph.item() == 4


def test_sidechain_dataset_opts_in_to_native_pickle_and_keeps_complete_residue(tmp_path):
    from graph_data import Data
    from utils.datasets import SidechainConformationDataset

    path = tmp_path / 'local_pdb.pkl'
    data = Data(atom_type=paddle.to_tensor([6, 6, 8, 6]),
                pos=paddle.to_tensor([[0., 0., 0.], [2., 0., 0.], [7., 0., 0.], [10., 0., 0.]]),
                edge_index=paddle.to_tensor([[0, 1, 2], [1, 0, 3]]),
                edge_type=paddle.to_tensor([1, 1, 1]),
                atom2res=paddle.to_tensor([0, 0, 1, 1]),
                is_sidechain=paddle.to_tensor([False, True, False, True]))
    with path.open('wb') as f:
        pickle.dump([data], f)
    with pytest.raises(ValueError, match=r'exported \.npz'):
        SidechainConformationDataset(path)
    dataset = SidechainConformationDataset(path, cutoff=0.5, fix_subgraph=False,
                                           trusted_pickle=True)
    # Either randomly chosen backbone atom selects both atoms in its residue.
    graph = dataset[0]
    assert graph.num_nodes == 2
    np.testing.assert_array_equal(graph.is_sidechain.numpy(), [False, True])
