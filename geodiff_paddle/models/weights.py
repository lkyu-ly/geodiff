"""Load model-only state with the release's native Paddle weight layout."""

import numpy as np
import paddle


def _array(value):
    return value.numpy() if isinstance(value, paddle.Tensor) else value


def set_model_state(model, state):
    """Validate every key, shape, dtype and alias before updating model tensors.

    State values may be native arrays or Paddle tensors. Shared model tensors
    must have identical values at every registered path, and are written once.
    """
    targets = model.state_dict()
    missing = targets.keys() - state.keys()
    unexpected = state.keys() - targets.keys()
    if missing or unexpected:
        raise ValueError(f'Model state keys: missing={sorted(missing)}, unexpected={sorted(unexpected)}')

    canonical = {}
    for key, target in targets.items():
        value = state[key]
        if not isinstance(value, (np.ndarray, paddle.Tensor)):
            raise TypeError(f'{key}: expected a NumPy array or Paddle tensor')
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f'{key}: shape {value.shape} != {target.shape}')
        expected_dtype = target.dtype
        if isinstance(value, np.ndarray):
            expected_dtype = str(target.dtype).removeprefix('paddle.')
        if value.dtype != expected_dtype:
            raise ValueError(f'{key}: dtype {value.dtype} != {target.dtype}')

        identity = id(target)
        if identity not in canonical:
            canonical[identity] = key, target, value
        else:
            first_key, _, first = canonical[identity]
            if value is not first and not np.array_equal(_array(first), _array(value)):
                raise ValueError(f'{key}: shared alias value disagrees with {first_key}')

    with paddle.no_grad():
        for _, target, value in canonical.values():
            target.set_value(value)


def load_released_weights(model, path):
    """Load a native .pdparams release file; its Linear weights are already transposed.

    The file holds the model state only. Keys, shapes, dtypes and shared aliases
    are validated before any model tensor is updated.
    """
    set_model_state(model, paddle.load(str(path)))
