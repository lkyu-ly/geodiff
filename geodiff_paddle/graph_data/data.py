"""GeoDiff subset of PyG 1.7.2 Data and Batch; see LICENSE and UPSTREAM."""

from copy import deepcopy
import re

import paddle


_INDEX = re.compile('index|face')


def _copy(value):
    if isinstance(value, paddle.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(item) for item in value)
    return deepcopy(value)


class Data:
    def __init__(self, num_nodes=None, **kwargs):
        self.__dict__.update(kwargs)
        if num_nodes is not None:
            self.num_nodes = num_nodes

    @property
    def keys(self):
        return [key for key, value in vars(self).items()
                if not key.startswith('_') and value is not None]

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __contains__(self, key):
        return key in self.keys

    def __iter__(self):
        return self()

    def __call__(self, *keys):
        for key in keys or sorted(self.keys):
            if self[key] is not None:
                yield key, self[key]

    def __cat_dim__(self, key, value):
        return -1 if _INDEX.search(key) else 0

    def __inc__(self, key, value):
        return self.num_nodes if _INDEX.search(key) else 0

    @property
    def num_nodes(self):
        if '_num_nodes' in vars(self):
            return self._num_nodes
        for key in ('atom_type', 'pos', 'x', 'batch'):
            if self[key] is not None:
                return self[key].shape[0]
        edge_index = self['edge_index']
        if edge_index is not None:
            return int(edge_index.max().item()) + 1 if edge_index.numel().item() else 0
        return None

    @num_nodes.setter
    def num_nodes(self, count):
        self._num_nodes = count

    @property
    def num_edges(self):
        return self.edge_index.shape[-1] if self['edge_index'] is not None else 0

    def clone(self):
        result = type(self).__new__(type(self))
        result.__dict__.update(_copy(vars(self)))
        return result

    def to(self, device, *keys):
        if isinstance(device, str) and device.startswith('cuda'):
            device = 'gpu' + device[4:]
        for key, item in self(*keys):
            if isinstance(item, paddle.Tensor):
                self[key] = item.to(device=device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=None):
        return self.to(f'gpu:{device}' if device is not None else 'gpu')


class Batch(Data):
    @classmethod
    def from_data_list(cls, data_list):
        batch = cls()
        batch._examples = [(type(data), vars(data).get('_num_nodes')) for data in data_list]
        counts = [data.num_nodes for data in data_list]
        batch.num_nodes = sum(counts)
        place = next((value.place for data in data_list for _, value in data
                      if isinstance(value, paddle.Tensor)), 'cpu')
        batch.batch = paddle.to_tensor([i for i, n in enumerate(counts) for _ in range(n)],
                                       dtype='int64', place=place)
        ptr = [0]
        for n in counts:
            ptr.append(ptr[-1] + n)
        batch.ptr = paddle.to_tensor(ptr, dtype='int64', place=place)
        batch._slices = {}
        for key in dict.fromkeys(key for data in data_list for key in data.keys):
            if key in ('batch', 'ptr'):
                raise ValueError(f'{key} is reserved for Batch')
            values = [data[key] for data in data_list]
            if not all(isinstance(value, paddle.Tensor) for value in values):
                batch[key] = values
                continue
            scalar = values[0].ndim == 0
            axis = 0 if scalar else data_list[0].__cat_dim__(key, values[0]) % values[0].ndim
            offset = start = 0
            chunks, slices = [], []
            for data, value in zip(data_list, values):
                chunk = value + offset if offset and value.dtype != paddle.bool else value
                chunk = chunk.unsqueeze(0) if scalar else chunk
                stop = start + chunk.shape[axis]
                chunks.append(chunk)
                slices.append((start, stop, offset))
                start = stop
                offset += data.__inc__(key, value)
            batch[key] = paddle.concat(chunks, axis=axis)
            batch._slices[key] = axis, scalar, slices
        return batch

    @property
    def num_graphs(self):
        return len(self._examples)

    def get_example(self, index):
        kind, num_nodes = self._examples[index]
        data = kind.__new__(kind)
        if num_nodes is not None:
            data.num_nodes = num_nodes
        for key, value in self:
            if key in ('batch', 'ptr'):
                continue
            if key in self._slices:
                axis, scalar, slices = self._slices[key]
                start, stop, offset = slices[index]
                selection = [slice(None)] * value.ndim
                selection[axis] = slice(start, stop)
                item = value[tuple(selection)]
                if offset and item.dtype != paddle.bool:
                    item = item - offset
                if scalar:
                    item = item.squeeze(0)
            else:
                item = value[index]
            if item is not None:
                data[key] = item
        return data

    def to_data_list(self):
        return [self.get_example(i) for i in range(self.num_graphs)]
