import argparse
import os
import pickle
from pathlib import Path

import paddle
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm

from graph_data import Compose
from models.epsnet import get_model
from models.weights import load_released_weights, set_model_state
from utils.datasets import PackedConformationDataset
from utils.misc import get_logger, get_new_log_dir, repeat_data, seed_all
from utils.transforms import AddHigherOrderEdges, CountNodesPerGraph


DEFAULT_CONFIG = Path(__file__).with_name('configs') / 'qm9_default.yml'


def num_confs(num: str):
    if num.endswith('x'):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('ckpt', nargs='?', type=Path, help='trusted native Paddle .pdckpt')
    source.add_argument('--weights', type=Path, help='released weights exported to NPZ')
    parser.add_argument('--config', type=Path, help='model and dataset YAML (default for NPZ: QM9)')
    parser.add_argument('--save_traj', action='store_true', default=False,
        help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, help='trusted local sampling results .pkl')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, help='exported test graph dataset .npz')
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, help='directory for local sampling results')
    parser.add_argument('--device', type=str, default='gpu:0')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000, help=
        'sampling num steps; for DSM framework, this means num steps for each noise scale'
        )
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
        help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0, help=
        'weight for global gradients')
    parser.add_argument('--sampling_type', type=str, default='ld', help=
        'generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics'
        )
    parser.add_argument('--eta', type=float, default=1.0, help=
        'weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()
    if args.ckpt is not None and args.ckpt.suffix != '.pdckpt':
        parser.error('expected a native .pdckpt; export released weights and use --weights instead')
    if args.device.startswith('cuda'):
        args.device = 'gpu' + args.device[4:]
    paddle.set_device(args.device)
    ckpt = paddle.load(str(args.ckpt)) if args.ckpt is not None else None
    if args.config is None and ckpt is not None and 'config' in ckpt:
        config = EasyDict(ckpt['config'])
    else:
        config_path = args.config
        if config_path is None:
            if ckpt is None:
                config_path = DEFAULT_CONFIG
            else:
                candidates = sorted(args.ckpt.parent.parent.glob('*.yml'))
                if not candidates:
                    parser.error('checkpoint has no config; pass --config')
                config_path = candidates[0]
        with config_path.open() as f:
            config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)
    if args.out_dir is None:
        log_dir = args.ckpt.parent.parent if args.ckpt is not None else args.weights.parent
        output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    else:
        output_dir = args.out_dir
        os.makedirs(output_dir, exist_ok=True)
    logger = get_logger('test', output_dir)
    logger.info(args)
    logger.info('Loading datasets...')
    transforms = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=config.model.edge_order)])
    test_set = PackedConformationDataset(args.test_set or config.dataset.test, transform=transforms)
    logger.info('Loading model...')
    model = get_model(config.model)
    if ckpt is None:
        load_released_weights(model, args.weights)
    else:
        set_model_state(model, ckpt['model'])
    model.eval()
    test_set_selected = [test_set[i] for i in
                         range(max(0, args.start_idx), min(len(test_set), args.end_idx))]
    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)
    for i, data in enumerate(tqdm(test_set_selected)):
        if data.smiles in done_smiles:
            logger.info('Molecule#%d is already done.' % i)
            continue
        num_samples = args.num_confs(data.num_pos_ref.item())
        data_input = data.clone()
        data_input['pos_ref'] = None
        batch = repeat_data(data_input, num_samples).to(args.device)
        clip_local = None
        for _ in range(2):
            try:
                pos_init = paddle.randn([batch.num_nodes, 3], device=args.device)
                pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type, pos_init=pos_init,
                    bond_index=batch.edge_index, bond_type=batch.edge_type,
                    batch=batch.batch, num_graphs=batch.num_graphs,
                    extend_order=False, n_steps=args.n_steps, step_lr=1e-06,
                    w_global=args.w_global, global_start_sigma=args.
                    global_start_sigma, clip=args.clip, clip_local=
                    clip_local, sampling_type=args.sampling_type, eta=args.eta)
                pos_gen = pos_gen.cpu()
                if args.save_traj:
                    data.pos_gen = paddle.stack(pos_gen_traj)
                else:
                    data.pos_gen = pos_gen
                results.append(data)
                done_smiles.add(data.smiles)
                save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                logger.info('Saving samples to: %s' % save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)
                break
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')
    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)

    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1
    results.sort(key=get_mol_key)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
