"""Compare a Torch L1 replay against a Paddle one with NumPy on CPU.

Both archives hold the same arrays: per-atom losses, the edges of the loss
forward, the recorded loss mean and pre-clip gradient norm, one gradient per
trained parameter and the names of the parameters without a gradient. Edges
and masks must be exactly equal; losses, both scalars and gradients pass at
atol=1e-4, rtol=1e-3. Each side must record without a gradient exactly the
five edge_encoder_local.* parameters, which the forward never calls. Exits
non-zero on any failure.
"""

import argparse
import sys

import numpy as np


ATOL, RTOL = 1e-4, 1e-3
# Denominator floor of the relative error, far below any float32 value here.
TINY = 1e-30
EXACT = ('edge_index', 'edge_type', 'local_edge_mask')
CLOSE = ('loss', 'loss_global', 'loss_local', 'edge_length', 'loss_mean', 'grad_norm')
SCALARS = ('loss_mean', 'grad_norm')
NO_GRAD_PARAMETERS = (
    'edge_encoder_local.bond_emb.weight',
    'edge_encoder_local.mlp.layers.0.weight',
    'edge_encoder_local.mlp.layers.0.bias',
    'edge_encoder_local.mlp.layers.1.weight',
    'edge_encoder_local.mlp.layers.1.bias',
)
DTYPES = dict(edge_index='i', edge_type='i', local_edge_mask='b', num_nodes='i',
              edge_length='f', loss='f', loss_global='f', loss_local='f',
              loss_mean='f', grad_norm='f')
GRAD = 'grad::'


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def stats(reference, actual):
    """Errors of `actual` against `reference`, both promoted to float64.

    max_rel_masked counts only entries above the absolute tolerance, where
    the relative error alone decides; max_rel counts every entry against
    max(|reference|, |actual|, TINY), so opposite small signs are not zero.
    """
    reference, actual = reference.astype(np.float64), actual.astype(np.float64)
    error = actual - reference
    scale = np.maximum(np.abs(reference), np.abs(actual))
    relative = np.abs(error) / np.maximum(scale, TINY)
    masked = relative[scale > ATOL]
    return dict(
        shape=str(reference.shape), max_abs=np.abs(error).max(), mean_abs=np.abs(error).mean(),
        max_rel_masked=float(masked.max()) if masked.size else 0.0,
        max_rel=float(relative.max()),
        norm_ratio=np.linalg.norm(actual) / max(np.linalg.norm(reference), 1e-300),
        passed=bool(np.allclose(actual, reference, atol=ATOL, rtol=RTOL)),
    )


def grad_norm(result):
    return np.sqrt(sum(float(np.square(result[name].astype(np.float64)).sum())
                       for name in result if name.startswith(GRAD)))


def no_grad_names(result, side):
    """Sorted names recorded in `no_grad`, or a failure if the array is not names."""
    if 'no_grad' not in result or result['no_grad'].dtype.kind not in 'US':
        return [], [f'{side} archive has no no_grad array of names']
    return sorted(str(name) for name in result['no_grad']), []


def compare(torch, paddle):
    """Return the report lines and the failures of `paddle` against `torch`."""
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
                        f'{result["mean_abs"]:>11.3e}{result["max_rel_masked"]:>11.3e}'
                        f'{result["max_rel"]:>11.3e}{result["norm_ratio"]:>11.6f}'
                        f'  {"yes" if result["passed"] else "NO"}')
            if not result['passed']:
                failures.append(f'{name}: outside atol={ATOL}, rtol={RTOL}')
        elif int(reference) != int(actual):
            failures.append(f'num_nodes: {int(reference)} vs {int(actual)}')

    gradients = [name for name in torch if name.startswith(GRAD) and name in paddle]
    torch_no_grad, torch_errors = no_grad_names(torch, 'torch')
    paddle_no_grad, paddle_errors = no_grad_names(paddle, 'paddle')
    failures += torch_errors + paddle_errors
    expected = sorted(NO_GRAD_PARAMETERS)
    if torch_no_grad != expected:
        failures.append(f'torch no_grad {torch_no_grad} is not exactly the 5 edge_encoder_local.* tensors')
    if paddle_no_grad != expected:
        failures.append(f'paddle no_grad {paddle_no_grad} is not exactly the 5 edge_encoder_local.* tensors')
    for side, names, result in (('torch', torch_no_grad, torch), ('paddle', paddle_no_grad, paddle)):
        recorded = [name for name in names if GRAD + name in result]
        if recorded:
            failures.append(f'{side} recorded gradients for parameters listed in no_grad: {recorded}')

    lines = [f'{"array":<56} {"shape":<16} {"max abs":>11} {"mean abs":>11} '
             f'{"max rel masked":>14} {"max rel":>11} {"norm ratio":>11}  ok', *rows]
    for name in CLOSE:
        if name in torch and name in paddle and torch[name].shape == paddle[name].shape:
            lines.append(f'{name} mean: torch {torch[name].astype(np.float64).mean():.10g}, '
                         f'paddle {paddle[name].astype(np.float64).mean():.10g}')
    for name in SCALARS:
        if name in torch and name in paddle:
            lines.append(f'{name}: torch {float(torch[name]):.10g}, paddle {float(paddle[name]):.10g}')
    torch_norm, paddle_norm = grad_norm(torch), grad_norm(paddle)
    lines.append(f'grad norm from saved gradients: torch {torch_norm:.10g}, paddle {paddle_norm:.10g}, '
                 f'ratio {paddle_norm / torch_norm:.8f}')
    lines.append(f'no_grad ({len(expected)}) required on both sides, '
                 f'torch and paddle identical: {torch_no_grad == paddle_no_grad}: {expected}')
    lines += [f'FAIL {failure}' for failure in failures]
    lines.append(f'{"PASS" if not failures else "FAIL"}: {len(failures)} failing check(s) '
                 f'over {len(gradients)} gradients, atol={ATOL}, rtol={RTOL}')
    return lines, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch', required=True, help='reference NPZ written by tools/replay_torch_step.py')
    parser.add_argument('--paddle', required=True, help='NPZ written by tools/replay_paddle_step.py')
    args = parser.parse_args(argv)
    lines, failures = compare(load(args.torch), load(args.paddle))
    print('\n'.join(lines))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
