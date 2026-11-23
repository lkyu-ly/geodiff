"""Export trusted GeoDiff graph datasets to compressed NPZ (offline, CPU only)."""

import argparse
from io import BytesIO
from pathlib import Path
import pickle
import sys

import numpy as np
import torch


TORCH_ROOT = Path(__file__).resolve().parents[1] / 'geodiff'


class _CPUStorage:
    def find_class(self, module, name):
        if (module, name) == ('torch.storage', '_load_from_bytes'):
            return self._load_storage
        return super().find_class(module, name)

    @staticmethod
    def _load_storage(buffer):
        return torch.load(BytesIO(buffer), map_location='cpu', weights_only=False)


class _CPUUnpickler(_CPUStorage, pickle.Unpickler):
    pass


class _PrefixFound(Exception):
    pass


class _GraphPrefixUnpickler(_CPUStorage, pickle._Unpickler):
    """Stop at a top-level list append once a requested prefix is available.

    Pickle emits batches of up to 1000 list entries. Decode only the first
    needed batches instead of holding the entire official training split.
    The Python implementation exposes opcode dispatch; the C one does not.
    """

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, file, limit):
        super().__init__(file)
        self.limit = limit

    def load(self):
        try:
            return super().load()[:self.limit]
        except _PrefixFound as prefix:
            return prefix.args[0]

    def load_append(self):
        super().load_append()
        self._check_prefix()

    def load_appends(self):
        super().load_appends()
        self._check_prefix()

    def _check_prefix(self):
        if not self.metastack and len(self.stack) == 1:
            root = self.stack[0]
            if type(root) is list and len(root) >= self.limit:
                raise _PrefixFound(root[:self.limit])

    dispatch[pickle.APPEND[0]] = load_append
    dispatch[pickle.APPENDS[0]] = load_appends


def _array(tensor):
    return tensor.detach().cpu().numpy()


def _pack(arrays, axis):
    if len({array.dtype for array in arrays}) != 1:
        raise ValueError('Cannot pack mixed dtypes for the same graph field')
    if axis is None:
        return np.stack(arrays)
    return np.concatenate(arrays, axis=axis)


def _ptr(sizes):
    return np.cumsum([0, *sizes], dtype=np.int64)


def _graph_arrays(path, limit):
    with path.open('rb') as file:
        reader = _CPUUnpickler(file) if limit is None else _GraphPrefixUnpickler(file, limit)
        graphs = reader.load()
    if not graphs:
        raise ValueError('No graphs to export')
    result = {
        'node_ptr': _ptr(graph.atom_type.shape[0] for graph in graphs),
        'edge_ptr': _ptr(graph.edge_index.shape[1] for graph in graphs),
    }
    for key, axis in (('atom_type', 0), ('pos', 0), ('edge_index', 1), ('edge_type', 0)):
        result[key] = _pack([_array(getattr(graph, key)) for graph in graphs], axis)
    for key in ('idx', 'totalenergy', 'boltzmannweight', 'smiles'):
        values = [getattr(graph, key, None) for graph in graphs]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f'Inconsistent optional graph field: {key}')
        result[key] = (np.asarray(values, dtype=np.str_) if key == 'smiles'
                       else _pack([_array(value) for value in values], None))
    if any(getattr(graph, 'rdmol', None) is not None for graph in graphs):
        molecules = [graph.rdmol.ToBinary() if getattr(graph, 'rdmol', None) is not None
                     else b'' for graph in graphs]
        result['rdmol_ptr'] = _ptr(map(len, molecules))
        result['rdmol_data'] = np.frombuffer(b''.join(molecules), dtype=np.uint8)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    graphs = commands.add_parser('graphs', help='export a trusted PyG graph-list pickle')
    graphs.add_argument('source', type=Path)
    graphs.add_argument('target', type=Path)
    graphs.add_argument('--limit', type=int, help='export only the first N graphs')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.source.resolve() == args.target.resolve() or (
            args.target.exists() and args.source.samefile(args.target)):
        parser.error('source and target must be different files')
    sys.path.insert(0, str(TORCH_ROOT))
    arrays = _graph_arrays(args.source, args.limit)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    with args.target.open('wb') as output:
        np.savez_compressed(output, **arrays)


if __name__ == '__main__':
    main()
