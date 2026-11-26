# GeoDiff on Paddle

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MinkaiXu/GeoDiff/blob/main/LICENSE)

[[OpenReview](https://openreview.net/forum?id=PzcvxEMzvQC)] [[arXiv](https://arxiv.org/abs/2203.02923)] [[Code](https://github.com/MinkaiXu/GeoDiff)]

This directory ports [GeoDiff](https://github.com/MinkaiXu/GeoDiff) (ICLR 2022) to Paddle. The [Torch reference](../geodiff/) is separate. Run the commands below from the repository root; dataset paths in YAML are relative to that directory.

## Set up Paddle

Use Python 3.10 and Paddle 3.4.0. For the Linux x86-64 CUDA 13 build used in the QM9 checks:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install 'setuptools==80.10.1' 'numpy==1.26.4' 'rdkit==2024.9.1' pyyaml easydict networkx tqdm
uv pip install https://paddle-whl.cdn.bcebos.com/stable/cu130/paddlepaddle-gpu/paddlepaddle_gpu-3.4.0-cp310-cp310-linux_x86_64.whl
```

Do not install Torch or PyG in the Paddle environment. `easydict` is a direct dependency of the Paddle entry scripts (`train.py`, `test.py`, the evaluators and the replay and conversion tools); install it explicitly, as above, because no other package pulls it in. Install `pandas` for the COV/MAT report, `psikit` and its Psi4 backend for property evaluation, or `py3Dmol` for interactive visualization. These are not needed for QM9 training. On WSL, add `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH` if the CUDA driver is not found.

## Export trusted QM9 assets offline

Use a separate Python 3.10 environment for the trusted Torch exports. The reference ran with Torch 2.11.0+cu128, torchvision 0.26.0+cu128, PyG 1.7.2, and the matching Linux x86-64 PyG extension wheels. Use the pinned binaries, not same-version source builds: CUDA radius neighbor selection differs between the `torch-cluster` wheel and source tag.

```bash
deactivate  # if the Paddle environment is active
uv venv --python 3.10 .venv-torch
source .venv-torch/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
uv pip install 'setuptools==80.10.1' 'numpy==1.26.4' 'rdkit==2024.9.1' \
  'torch-geometric==1.7.2' 'googledrivedownloader==1.1.0' pyyaml easydict
uv pip install --no-index --no-deps --find-links 'https://data.pyg.org/whl/torch-2.11.0%2Bcu128.html' \
  'torch-scatter==2.1.2+pt211cu128' 'torch-sparse==0.6.18+pt211cu128' 'torch-cluster==1.6.3+pt211cu128'
```

Place the [released checkpoint and preprocessed GEOM splits](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing) at `checkpoints/qm9_default.pt` and `data/QM9/*.pkl`. The weight converter imports Paddle next to Torch, so install the CPU Paddle wheel of the same 3.4.0 version into this environment as well. Do not use the CUDA 13 wheel from [Set up Paddle](#set-up-paddle): the Torch cu128 and Paddle cu130 wheels pin conflicting `cuda-toolkit` versions, and the converter runs on CPU anyway.

```bash
uv pip install --find-links https://www.paddlepaddle.org.cn/packages/stable/cpu/ 'paddlepaddle==3.4.0'
```

Run the following commands from the repository root in the Torch environment:

```bash
python tools/convert_released_weights.py checkpoints/qm9_default.pt runs/paddle_qm9/qm9_default.pdparams
python tools/export_paddle_assets.py graphs data/QM9/train_data_40k.pkl runs/paddle_qm9/train_2.npz --limit 2
python tools/export_paddle_assets.py graphs data/QM9/val_data_5k.pkl runs/paddle_qm9/val_2.npz --limit 2
```

Both tools read trusted files only, run on CPU, and leave the originals unchanged. The `.pdparams` holds the model weights in Paddle layout and nothing else: no config, iteration, Adam or scheduler state. The converter transposes Linear weights only and checks every saved entry against the checkpoint; Paddle loads the file with `paddle.load`. Graph NPZs keep graph order, molecule-local edge indices, and available SMILES and RDKit molecule data. Paddle reads those archives without pickle. For a complete split, omit `--limit` and write the corresponding NPZ at the path listed in the config.

## Compare fixed QM9 forward outputs

To recreate both references after cleaning `runs/`, run this command in the Torch environment on CUDA:

```bash
python tools/export_qm9_references.py \
  --checkpoint checkpoints/qm9_default.pt --val data/QM9/val_data_5k.pkl \
  --forward runs/paddle_qm9/recreated/forward.npz \
  --val-graph runs/paddle_qm9/recreated/val_first_graph.npz
```

The tool loads the checkpoint on CPU, then runs the original model on CUDA with the fixed seed-2021 NumPy inputs. It uses the original model's graph expansion on the first official validation graph. It writes only new files. The recreated forward archive is a new execution, not the historical `runs/torch_qm9_smoke/forward.npz`; keep the historical file unchanged if you have it.

Return to the Paddle environment and check the archives:

```bash
PYTHONPATH=geodiff_paddle python tools/check_paddle_forward.py \
  --forward runs/paddle_qm9/recreated/forward.npz \
  --val-graph runs/paddle_qm9/recreated/val_first_graph.npz \
  --weights runs/paddle_qm9/qm9_default.pdparams \
  --train runs/paddle_qm9/train_2.npz --device gpu:0
```

This checks edges, types and masks exactly, and float32 lengths and scores at `atol=1e-5, rtol=1e-4`. It also checks loss and backward on the exported training graphs, without an optimizer step.


## Compare one training step with Torch

Both frameworks replay the same recorded step with the released weights and no optimizer update. Each saves the per-atom losses, the edges of its loss forward, the loss mean, the pre-clip gradient norm, one gradient per trained parameter and the names of the parameters without a gradient. The step NPZ comes from the Torch training run started with `--record_inputs`, which writes `inputs/<iteration>.npz` below the run directory. The Torch tool adds `geodiff/` to its own import path.

In the Torch environment on CUDA:

```bash
python tools/replay_torch_step.py --checkpoint checkpoints/qm9_default.pt \
  --inputs runs/torch_qm9_smoke/qm9_smoke_2026_11_23__20_27_55_resume/inputs/2865000.npz \
  --out runs/l1_alignment/torch_step2865000_replay.npz
```

To export the official conformations of that batch, run the same command with `--checkpoint` and `--out` replaced by `--graphs-source data/QM9/train_data_40k.pkl --graphs-out runs/l1_alignment/step2865000_graphs_official.npz`. It finds each recorded graph in the split by atom types, positions, bond indices and bond types, asserts that atoms, bonds and types equal the recording, and writes them in the layout `graph_data.load_graphs` reads, with the matching dataset positions.

In the Paddle environment:

```bash
PYTHONPATH=geodiff_paddle python tools/replay_paddle_step.py \
  --weights runs/paddle_qm9/qm9_default.pdparams \
  --inputs runs/torch_qm9_smoke/qm9_smoke_2026_11_23__20_27_55_resume/inputs/2865000.npz \
  --out runs/l1_alignment/paddle_step2865000_replay.npz --device gpu:0
```

Adding `--graphs-out FILE` here only repacks the recorded arrays in that layout, without reading the dataset. Use the Torch export above when the graphs must come from the official split.

Then compare the two archives with NumPy only, in either environment:

```bash
python tools/compare_l1_alignment.py \
  --torch runs/l1_alignment/torch_step2865000_replay.npz \
  --paddle runs/l1_alignment/paddle_step2865000_replay.npz
```

Edges, edge types and the local mask must be exactly equal. Losses, both scalars and gradients pass at `atol=1e-4, rtol=1e-3`; the relative error is reported over all entries and over those above `atol`. Both sides must record without a gradient exactly the five `edge_encoder_local.*` tensors, which the forward never calls. The command exits non-zero on any failure.


## Compare fixed-input forward and 20 training updates

Run from the repository root in the Torch environment. Choose a fresh `run` directory and point `step` at the saved first training batch:

```bash
run=runs/qm9-end-to-end/replay
step=runs/torch_qm9_smoke/qm9_smoke_2026_11_23__20_27_55_resume/inputs/2865000.npz
```

If the recording was cleaned up, capture a new one from the released checkpoint and official QM9 splits. This runs one Torch update and one validation batch. Both sides will replay the new recording; it does not replace the historical file.

```bash
mkdir -p "$run"
capture=$(mktemp -d "$run/torch-capture.XXXXXX")
mkdir -p "$capture/resume/checkpoints"
ln -s "$PWD/checkpoints/qm9_default.pt" "$capture/resume/checkpoints/2865000.pt"
python - "$capture/resume/qm9.yml" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path('geodiff/configs/qm9_default.yml').read_text())
config['train']['max_iters'] = 2865000
config['dataset'].update(train='data/QM9/train_data_40k.pkl',
                         val='data/QM9/val_data_5k.pkl')
Path(sys.argv[1]).write_text(yaml.safe_dump(config, sort_keys=False))
PY
python geodiff/train.py "$capture/resume" --device cuda:0 --resume_iter 2865000 \
  --record_inputs --trust_checkpoint --val_batches 1 --logdir "$capture/results"
step=$(find "$capture/results" -type f -path '*/inputs/2865000.npz' -print -quit)
test -s "$step"
```

Generate the fixed-forward reference and run the Torch training replay:

```bash
python tools/export_qm9_references.py \
  --checkpoint checkpoints/qm9_default.pt --forward "$run/forward.npz"
python tools/run_qm9_training_parity.py prepare \
  --inputs "$step" --out "$run/train_inputs.npz"
python tools/run_qm9_training_parity.py torch \
  --checkpoint checkpoints/qm9_default.pt --inputs "$run/train_inputs.npz" \
  --out "$run/train_torch.json"
```

In the Torch-free Paddle environment, with `run` pointing to the same directory:

```bash
run=runs/qm9-end-to-end/replay
PYTHONPATH=geodiff_paddle python tools/check_paddle_forward.py \
  --forward "$run/forward.npz" --weights runs/paddle_qm9/qm9_default.pdparams
python tools/run_qm9_training_parity.py paddle \
  --weights runs/paddle_qm9/qm9_default.pdparams --inputs "$run/train_inputs.npz" \
  --out "$run/train_paddle.json"
python tools/run_qm9_training_parity.py compare \
  --torch "$run/train_torch.json" --paddle "$run/train_paddle.json" \
  --out "$run/train_comparison.json"
```

The forward check compares graph indices, types and masks exactly, and checks edge lengths and both scores with `atol=1e-5, rtol=1e-4`. Training reuses the recorded batch of 64 conformations. Both frameworks use its original first-step noise and the same 19 further timestep/noise pairs prepared with NumPy seed 2021. Each starts from the released weights with two new Adam optimizers at the QM9 learning rate, with no validation or scheduler updates during the replay.

Each training command writes its step JSON and an adjacent `.json.inputs.npz` containing the inputs it consumed. Keep each pair together; comparison requires the expected backend labels and identical input arrays.

The comparison reports absolute and relative errors for total, global and local loss. Relative error uses `max(abs(Torch), abs(Paddle), 1e-30)` as denominator. The training verdict checks whether the total-loss differences `Paddle − Torch` include both signs. The report is written even if this criterion fails.

## Run one training update and resume

In the Paddle environment, make a two-graph batch config and run one update followed by validation:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path('geodiff_paddle/configs/qm9_default.yml').read_text())
cfg['dataset'].update(train='runs/paddle_qm9/train_2.npz', val='runs/paddle_qm9/val_2.npz')
cfg['train']['batch_size'] = 2
Path('runs/paddle_qm9/qm9_smoke.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

python geodiff_paddle/train.py runs/paddle_qm9/qm9_smoke.yml \
  --device gpu:0 --weights runs/paddle_qm9/qm9_default.pdparams \
  --max_iters 1 --val_batches 1 --logdir runs/paddle_qm9/train

run=$(ls -dt runs/paddle_qm9/train/qm9_smoke_* | head -n 1)
python geodiff_paddle/train.py "$run" \
  --device gpu:0 --resume_iter 1 --max_iters 2 --val_batches 1 --logdir runs/paddle_qm9/train
```

The fresh run starts with released model weights and two new Adam optimizers; the resume run restores model and optimizer state from the trusted native `checkpoints/1.pdckpt`. This checks one update and native resume, not historical Torch training.

## Run the tests

In the Paddle environment, install `pytest` with `uv pip install pytest` and run `pytest` from the repository root. The portable reader, comparator and Paddle replay tests need no Torch or official assets. Set `GEODIFF_TORCH_PYTHON` to an interpreter that imports Torch, as in the conversion environment above, to include the exporter round-trip; when that interpreter also imports Paddle, the converter round-trip runs there too. Without the variable, the converter round-trip runs only if the test process itself imports Torch, and is skipped elsewhere. Set `GEODIFF_QM9_SMOKE_ASSETS` to `runs/paddle_qm9` to include the bounded CUDA training and native resume test on the exported two-graph splits and the converted `qm9_default.pdparams`.

## Optional sampling and evaluation

In the offline Torch environment, export the complete test and property splits. Do not pass the official PyG pickles directly to any Paddle entry.

```bash
python tools/export_paddle_assets.py graphs data/QM9/test_data_1k.pkl data/QM9/test_data_1k.npz
python tools/export_paddle_assets.py graphs data/QM9/qm9_property.pkl data/QM9/qm9_property.npz
```

In the Paddle environment, sample with released weights or replace `--weights FILE.pdparams` with a trusted native positional `ITER.pdckpt`. Released weights use the QM9 config by default; `--config` selects a different YAML. Native checkpoints use their embedded config unless overridden. `--test_set` overrides the dataset path; use the property NPZ there for property evaluation.

```bash
python geodiff_paddle/test.py --weights runs/paddle_qm9/qm9_default.pdparams \
  --start_idx 0 --end_idx 5 --out_dir runs/paddle_qm9/sample
python geodiff_paddle/eval_covmat.py.py runs/paddle_qm9/sample/samples_all.pkl

python geodiff_paddle/test.py --weights runs/paddle_qm9/qm9_default.pdparams \
  --test_set data/QM9/qm9_property.npz --num_confs 50 \
  --start_idx 0 --end_idx 5 --out_dir runs/paddle_qm9/property
python geodiff_paddle/eval_prop.py --dataset data/QM9/qm9_property.npz \
  --generated runs/paddle_qm9/property/samples_all.pkl
```

Sampling groups reference conformers by SMILES in archive order. The `samples_*.pkl` files store local Paddle graph objects and RDKit molecules; use only trusted local files for sampling `--resume` and evaluator input. COV/MAT writes `_covmat.csv` and `_covmat.pkl`. The property evaluator keeps separate `_ref_prop.pkl` and `_gen_prop.pkl` caches beside their inputs; load only trusted native caches. Optional Python dataset classes can open locally preprocessed native graphs via `trusted_pickle=True`; NPZ remains the default and no loader falls back to official PyG pickle.

The optional entry checks cover CLI startup, weight and checkpoint loading without a sampling step, graph preparation, and the two COV/MAT reduction axes. They do not validate complete diffusion sampling, Drugs, property estimates, or paper metrics. For PyMol usage, see the [upstream guide](../geodiff/README.md#visualizing-molecules-with-pymol).

## Citation
If you use GeoDiff, please cite the original paper:
```
@inproceedings{
xu2022geodiff,
title={GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation},
author={Minkai Xu and Lantao Yu and Yang Song and Chence Shi and Stefano Ermon and Jian Tang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=PzcvxEMzvQC}
}
```

## Acknowledgement

GeoDiff builds on the [ConfGF codebase](https://github.com/DeepGraphLearning/ConfGF).