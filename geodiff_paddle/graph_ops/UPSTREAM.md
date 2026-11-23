# Graph operator sources

These operators cover the tensor forms used by GeoDiff, not the complete upstream libraries. All five downloaded distributions below contain the same MIT license text, retained in [LICENSE](LICENSE).

| Dependency | Checked source |
| --- | --- |
| PyTorch Geometric | [1.7.2 source tag](https://github.com/pyg-team/pytorch_geometric/tree/1.7.2) |
| torch-scatter | [2.1.2 source tag](https://github.com/rusty1s/pytorch_scatter/tree/2.1.2) |
| torch-sparse | [0.6.18 source tag](https://github.com/rusty1s/pytorch_sparse/tree/0.6.18) |
| torch-cluster | [1.6.3 source tag](https://github.com/rusty1s/pytorch_cluster/tree/1.6.3) and [1.6.3+pt211cu128 Linux CPython 3.10 wheel](https://data.pyg.org/whl/torch-2.11.0%2Bcu128/torch_cluster-1.6.3%2Bpt211cu128-cp310-cp310-linux_x86_64.whl) |

## From the trimmed Python to this package

The source tags and matching cluster wheel were downloaded and extracted before conversion. `trim.py` selected ten Python files by function or method, removed docstrings, and kept only used reduction branches. The extracted wheel `torch_cluster/radius.py` was compared byte-for-byte with the reference environment's wrapper and used as the converter input instead of the tag's wrapper.

| Python functions retained for PaConvert | Native Paddle result |
| --- | --- |
| PyG `torch_geometric/utils/to_dense_adj.py:to_dense_adj`; `torch_geometric/utils/sparse.py:dense_to_sparse` | `adjacency.py` keeps duplicate accumulation into flattened adjacency, row-major nonzero extraction and batched offsets. It uses native `index_add` and `gather_nd` and handles explicit graph size and empty edges. |
| scatter `torch_scatter/scatter.py:scatter_sum,scatter_add,scatter_mean,scatter_max,scatter`; `torch_scatter/utils.py:broadcast` | `reduction.py` keeps explicit or inferred output size, count-clamped means and integer floor division. `index_add` takes the one-dimensional index directly; no expansion or monkey-patched `scatter_add_` is needed. `put_along_axis` and a gather replace the compiled max; ties route gradient to one winning input. |
| sparse `torch_sparse/coalesce.py:coalesce`; `torch_sparse/storage.py:SparseStorage.__init__,row,col,value,coalesce`; `torch_sparse/utils.py:index_sort`; scatter `torch_scatter/segment_csr.py:segment_csr,segment_sum_csr,segment_add_csr` | `adjacency.py:coalesce` keeps the `row * n + col` key and duplicate sum. Sorted `unique` and the inverse index replace the storage/sort/CSR backend; a summed value of zero does not drop its coordinate. The optional CSR-to-COO path is not needed by a COO input. |
| PyG `torch_geometric/nn/pool/__init__.py:radius,radius_graph`; wheel `torch_cluster/radius.py:radius,radius_graph` | `radius.py` keeps the wrapper's raw `(query, source)` and default graph `(source, target)` orientation, replaces the opaque extension with a Paddle squared-distance mask and first-match cap, and partitions sources by their sorted batch IDs. |

PaConvert **3.3.1** reported **225/230** converted calls (**97.83%**). Its five unsupported calls were `torch.ops.torch_scatter.segment_sum_csr` (twice), `torch.ops.torch_scatter.scatter_max`, `torch.ops.torch_sparse.ptr2ind`, and `torch.ops.torch_cluster.radius`. These correspond to the extension entry points in scatter `csrc/segment_csr.cpp` and `csrc/scatter.cpp`, sparse `csrc/convert.cpp`, and cluster `csrc/radius.cpp` plus its CUDA backend. The generated helpers still monkey-patched tensor methods and retained third-party imports; neither the helpers nor the extension dispatches are shipped here.

Local evidence is under `runs/third_party_qm9/01/`: extracted releases and wheel wrapper, `trim.py`, `trim.log`, `trimmed_torch/`, `converted_paddle/`, `paconvert.log`, and the direct `torch-radius.npz` CUDA oracle. The conversion command, run from that scratch directory, was:

```sh
paconvert -i "$PWD/trimmed_torch" -o "$PWD/converted_paddle" \
  --show_unsupport_api --no_format --log_dir "$PWD/paconvert.log"
```

## Radius cap in the checked CUDA wheel

The `1.6.3` tag requests 33 candidates before removing self-loops. The checked wheel's `radius.py:140–142` instead passes `max_num_neighbors=32` and `ignore_same_index=True` for the default graph. GeoDiff follows the wheel: exclude equal indices before capping, then take the first 32 sources in input order at squared distance strictly below `r²`. Coincident points with distinct indices still count. Raw sidechain `radius` leaves `ignore_same_index=False` and returns the first 32 sources for each query; sidechain reindexing and reverse-edge assembly are the caller's job.

On Torch **2.11.0+cu128**, PyG **1.7.2**, scatter **2.1.2+pt211cu128**, sparse **0.6.18+pt211cu128**, and cluster **1.6.3+pt211cu128**, 35 FP32 points `arange(35) / 100` at `r=1` gave 1,120 directed edges. Target 0 used sources `1..32`, target 34 used `0..31`, and raw queries at indices 0 and 34 both used `0..31` (64 edges). A separate small fixture included an exact-cutoff pair, distinct coincident indices, overlapping different batches, and an isolated node.

Paddle **3.4.0** passed all six graph-operator tests on CPU and CUDA. On CUDA, all four saved Torch edge arrays matched exactly: default radius graph `(2, 1120)`, raw sidechain `(2, 64)`, strict batched graph `(2, 10)`, and strict batched raw radius `(2, 5)`.
