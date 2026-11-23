"""Convert a trusted GeoDiff Torch checkpoint to native Paddle weights (offline, CPU only).

Run once where both Torch and Paddle import, for example in the Torch environment
with the Paddle wheel installed as well:

    python tools/convert_released_weights.py checkpoints/qm9_default.pt runs/paddle_qm9/qm9_default.pdparams
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import paddle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'geodiff_paddle'))

from models.weights import set_model_state  # noqa: E402


def convert(source, target, get_model):
    """Write the model weights of checkpoint `source` to native `target`.

    A skeleton from `get_model` supplies the keys, shapes and shared aliases of
    the Paddle layout. Only Linear weights are transposed. The output holds the
    model state and nothing else, and is verified after writing.
    """
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    with paddle.utils.unique_name.guard():
        skeleton = get_model(checkpoint['config']['model'])
    template = skeleton.state_dict()
    if template.keys() != state.keys():
        raise ValueError(f'Model key mismatch: missing={template.keys() - state.keys()}, '
                         f'extra={state.keys() - template.keys()}')
    linear_weights = {id(layer.weight) for layer in skeleton.sublayers()
                      if isinstance(layer, paddle.nn.Linear)}
    arrays, aliases = {}, {}
    for key, tensor in state.items():
        identity = id(template[key])
        # Linear weights swap their two axes between the Torch and Paddle layouts.
        shape = tuple(tensor.shape)[::-1] if identity in linear_weights else tuple(tensor.shape)
        if shape != tuple(template[key].shape):
            raise ValueError(f'{key}: expected shape {shape} in Torch layout, got {tuple(tensor.shape)}')
        previous = aliases.setdefault(identity, key)
        if previous != key and not torch.equal(state[previous], tensor):
            raise ValueError(f'Inconsistent checkpoint aliases: {previous} and {key}')
        array = tensor.detach().cpu().numpy()
        arrays[key] = array.T if identity in linear_weights else array

    target.parent.mkdir(parents=True, exist_ok=True)
    paddle.save({key: paddle.to_tensor(array) for key, array in arrays.items()}, str(target))
    saved = paddle.load(str(target))
    if saved.keys() != arrays.keys():
        raise ValueError(f'Saved key mismatch: {saved.keys() ^ arrays.keys()}')
    for key, array in arrays.items():
        if not np.array_equal(saved[key].numpy(), array):
            raise ValueError(f'{key}: saved weights differ from the checkpoint')
    with paddle.utils.unique_name.guard():
        set_model_state(get_model(checkpoint['config']['model']), saved)
    return len(arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='trusted GeoDiff Torch checkpoint')
    parser.add_argument('target', type=Path, help='native Paddle .pdparams to write')
    args = parser.parse_args()
    if args.source.resolve() == args.target.resolve() or (
            args.target.exists() and args.source.samefile(args.target)):
        parser.error('source and target must be different files')
    paddle.set_device('cpu')
    from models.epsnet import get_model

    count = convert(args.source, args.target, get_model)
    print(f'{args.target}: {count} model state entries written and verified against {args.source}')


if __name__ == '__main__':
    main()
