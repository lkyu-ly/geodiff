"""Compare a Torch L1 replay against a Paddle one with NumPy on CPU.

Edges and masks must be exactly equal; losses and gradients pass at
atol=1e-4, rtol=1e-3. Exits non-zero on any failure.

The Torch recording script saved an empty no_grad list by a dedup bug, so the
Torch no-gradient set is taken as the fixed five parameters listed in
NO_GRAD_PARAMETERS, which the Torch archive must not carry gradients for.
"""

import argparse
import sys

import numpy as np


ATOL, RTOL = 1e-4, 1e-3
EXACT = ('edge_index', 'edge_type', 'local_edge_mask')
CLOSE = ('edge_length', 'loss', 'loss_global', 'loss_local')
NO_GRAD_PARAMETERS = (
    'edge_encoder_local.bond_emb.weight',
    'edge_encoder_local.mlp.layers.0.weight',
    'edge_encoder_local.mlp.layers.0.bias',
    'edge_encoder_local.mlp.layers.1.weight',
    'edge_encoder_local.mlp.layers.1.bias',
)
DTYPES = dict(edge_index='i', edge_type='i', local_edge_mask='b', num_nodes='i',
              edge_length='f', loss='f', loss_global='f', loss_local='f')
GRAD = 'grad::'


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def stats(reference, actual):
    """Errors of `actual` against `reference`, both promoted to float64."""
    reference, actual = reference.astype(np.float64), actual.astype(np.float64)
    error = actual - reference
    # Relative error counts only entries above the absolute tolerance.
    scale = np.maximum(np.abs(reference), np.abs(actual))
    relative = np.abs(error) / np.where(scale > ATOL, scale, np.inf)
    return dict(
        shape=str(reference.shape), max_abs=np.abs(error).max(), mean_abs=np.abs(error).mean(),
        max_rel=relative.max(),
        norm_ratio=np.linalg.norm(actual) / max(np.linalg.norm(reference), 1e-300),
        passed=bool(np.allclose(actual, reference, atol=ATOL, rtol=RTOL)),
    )


def grad_norm(result):
    return np.sqrt(sum(float(np.square(result[name].astype(np.float64)).sum())
                       for name in result if name.startswith(GRAD)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch', required=True, help='reference NPZ recorded with Torch')
    parser.add_argument('--paddle', required=True, help='NPZ written by tools/replay_paddle_step.py')
    args = parser.parse_args()
    torch, paddle = load(args.torch), load(args.paddle)
    failures, rows = [], []

    if set(torch) != set(paddle):
        failures.append(f'array names differ: torch-only {sorted(set(torch) - set(paddle))}, '
                        f'paddle-only {sorted(set(paddle) - set(torch))}')
    for name, kind in DTYPES.items():
        if name not in torch or name not in paddle:
            failures.append(f'{name}: missing on one side')
        elif torch[name].dtype.kind != kind or paddle[name].dtype.kind != kind:
            failures.append(f'{name}: dtype {torch[name].dtype} vs {paddle[name].dtype}, expected {kind}')

    for name in sorted(set(torch) & set(paddle)):
        reference, actual = torch[name], paddle[name]
        if name == 'no_grad' or reference.dtype.kind not in 'fib':
            continue
        label = name.removeprefix(GRAD)
        if reference.shape != actual.shape:
            rows.append(f'{label:<56} {str(reference.shape) + " vs " + str(actual.shape):<16} SHAPE MISMATCH')
            failures.append(f'{name}: shape {reference.shape} vs {actual.shape}')
        elif name in EXACT:
            equal = np.array_equal(reference, actual)
            rows.append(f'{label:<56} {str(reference.shape) + " (exact)":<16}'
                        + ('equal' if equal else f'{np.count_nonzero(reference != actual)} entries differ'))
            if not equal:
                failures.append(f'{name}: not exactly equal')
        elif name != 'num_nodes':
            result = stats(reference, actual)
            rows.append(f'{label:<56} {result["shape"]:<16}{result["max_abs"]:>11.3e}'
                        f'{result["mean_abs"]:>11.3e}{result["max_rel"]:>11.3e}'
                        f'{result["norm_ratio"]:>11.6f}  {"yes" if result["passed"] else "NO"}')
            if not result['passed']:
                failures.append(f'{name}: outside atol={ATOL}, rtol={RTOL}')
        elif int(reference) != int(actual):
            failures.append(f'num_nodes: {int(reference)} vs {int(actual)}')

    gradients = [name for name in torch if name.startswith(GRAD) and name in paddle]
    no_grad = sorted(str(name) for name in paddle['no_grad'])
    if no_grad != sorted(NO_GRAD_PARAMETERS):
        failures.append(f'paddle no_grad {no_grad} is not exactly the 5 edge_encoder_local.* tensors')
    torch_no_grad = sorted(name for name in NO_GRAD_PARAMETERS if GRAD + name not in torch)
    if torch_no_grad != sorted(NO_GRAD_PARAMETERS):
        failures.append(f'torch recorded gradients for {sorted(set(NO_GRAD_PARAMETERS) - set(torch_no_grad))}')

    print(f'{"array":<56} {"shape":<16} {"max abs":>11} {"mean abs":>11} {"max rel":>11} {"norm ratio":>11}  ok')
    print('\n'.join(rows))
    for name in CLOSE[1:]:
        if name in torch and name in paddle and torch[name].shape == paddle[name].shape:
            print(f'{name} mean: torch {torch[name].astype(np.float64).mean():.10g}, '
                  f'paddle {paddle[name].astype(np.float64).mean():.10g}')
    torch_norm, paddle_norm = grad_norm(torch), grad_norm(paddle)
    print(f'pre-clip grad norm: torch {torch_norm:.10g}, paddle {paddle_norm:.10g}, '
          f'ratio {paddle_norm / torch_norm:.8f}')
    print(f'no_grad ({len(no_grad)}), identical on both sides: {no_grad == torch_no_grad}')
    for failure in failures:
        print(f'FAIL {failure}')
    print(f'{"PASS" if not failures else "FAIL"}: {len(failures)} failing check(s) '
          f'over {len(gradients)} gradients, atol={ATOL}, rtol={RTOL}')
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    main()
