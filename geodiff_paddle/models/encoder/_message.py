"""Adjacency conventions for the two local message reductions.

The gather/target-reduce logic in gin.py and schnet.py follows PyG 1.7.2
`torch_geometric/nn/conv/message_passing.py` (_lift/_collect/aggregate,
lines 124–184 and 270–288); MIT license: ../../graph_ops/LICENSE.
PaConvert 3.3.1 converted its trimmed dense branches before the native
adaptation. Local inputs, generated code and log: runs/third_party_qm9/03/.
"""


def sparse_coo(edge_index):
    """Return native transposed sparse adjacency as COO, or None for dense indices."""
    if edge_index.is_sparse_coo():
        return edge_index
    if edge_index.is_sparse_csr():
        return edge_index.to_sparse_coo(2)
    return None
