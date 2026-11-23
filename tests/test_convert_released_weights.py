"""One-time conversion of a trusted Torch checkpoint to native Paddle weights.

The Torch side runs in a subprocess, so the test process stays Torch-free. The
subprocess uses GEODIFF_TORCH_PYTHON, else the test interpreter if it imports
Torch; elsewhere the tests are skipped.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import paddle
import pytest
from paddle import nn

from models.weights import load_released_weights

ROOT = Path(__file__).resolve().parents[1]
STATE_KEYS = {'embed.weight', 'square.weight', 'square.bias', 'wide.weight', 'wide.bias',
              'alias.weight', 'alias.bias', 'scale'}

# An embedding, a square Linear, a shared Linear alias and a buffer. Asymmetric
# values expose a missing transpose, also on the square weight.
TORCH_CHECKPOINT = '''
import sys
from pathlib import Path
import numpy as np
import torch
from paddle import nn  # noqa: F401  checks that both frameworks import here
from tests.test_convert_released_weights import PaddleLayers
from tools.convert_released_weights import convert


class TorchLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(5, 3)
        self.square = torch.nn.Linear(3, 3)
        self.wide = torch.nn.Linear(4, 2)
        self.alias = self.wide
        self.register_buffer('scale', torch.tensor([2., 0.5]))


model = TorchLayers()
with torch.no_grad():
    model.square.weight.copy_(torch.arange(9, dtype=torch.float32).reshape(3, 3))
    model.wide.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4) - 4)
    model.square.bias.copy_(torch.tensor([1., -1., 0.5]))
    model.wide.bias.copy_(torch.tensor([-2., 3.]))
    model.embed.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
source, reference, target = map(Path, sys.argv[1:])
torch.save({'config': {'model': {}}, 'model': model.state_dict(), 'iteration': 7,
            'optimizer_global': {}, 'avg_val_loss': 1.5}, source)
np.savez(reference, **{key: value.numpy() for key, value in model.state_dict().items()})
convert(source, target, lambda config: PaddleLayers())
'''


SAME_FILE = '''
import os
import sys
from pathlib import Path
from tools.convert_released_weights import main

source, target = map(Path, sys.argv[1:])
os.link(source, target)  # a second name for the checkpoint file itself
before = source.read_bytes()
sys.argv = ['convert_released_weights.py', str(source), str(target)]
try:
    main()
except SystemExit as exit:
    assert exit.code == 2, exit.code
else:
    raise AssertionError('the converter accepted one file as source and target')
assert source.read_bytes() == before, 'the converter overwrote the checkpoint'
'''


class PaddleLayers(nn.Layer):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 3)
        self.square = nn.Linear(3, 3)
        self.wide = nn.Linear(4, 2)
        self.alias = self.wide
        self.register_buffer('scale', paddle.to_tensor([2., 0.5]))


def torch_python():
    """Interpreter of the conversion subprocess, which imports Torch next to Paddle."""
    python = os.environ.get('GEODIFF_TORCH_PYTHON')
    if python is None and importlib.util.find_spec('torch') is None:
        pytest.skip('Set GEODIFF_TORCH_PYTHON to an interpreter that imports Torch next to Paddle')
    return python or sys.executable


def run_torch(python, script, *paths):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(
        str(path) for path in (ROOT, ROOT / 'geodiff_paddle', os.environ.get('PYTHONPATH')) if path)}
    result = subprocess.run([python, '-c', script, *(str(path) for path in paths)],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def converted(tmp_path):
    source, reference, target = (tmp_path / name for name in
                                 ('released.pt', 'released.npz', 'released.pdparams'))
    run_torch(torch_python(), TORCH_CHECKPOINT, source, reference, target)
    return source, reference, target
def test_converted_weights_hold_only_model_entries_in_paddle_layout(converted):
    source, reference, target = converted

    saved = paddle.load(str(target))
    assert set(saved) == STATE_KEYS
    assert all(isinstance(value, paddle.Tensor) and value.dtype == paddle.float32
               for value in saved.values())
    with np.load(reference) as torch_state:
        assert set(torch_state.files) == STATE_KEYS
        for key in ('square.weight', 'wide.weight', 'alias.weight'):
            np.testing.assert_array_equal(saved[key].numpy(), torch_state[key].T, err_msg=key)
        for key in ('embed.weight', 'square.bias', 'wide.bias', 'alias.bias', 'scale'):
            np.testing.assert_array_equal(saved[key].numpy(), torch_state[key], err_msg=key)

    model = PaddleLayers()
    load_released_weights(model, target)
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), saved[key].numpy(), err_msg=key)


def test_converter_refuses_one_file_as_source_and_target(converted):
    source, _, _ = converted
    run_torch(torch_python(), SAME_FILE, source, source.with_name('linked.pdparams'))
