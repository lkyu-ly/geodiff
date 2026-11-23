import argparse
from itertools import islice
from pathlib import Path
import shutil

from easydict import EasyDict
import numpy as np
import paddle
from tqdm.auto import tqdm
import yaml

from graph_data import GraphDataLoader, GraphDataset
from models.epsnet import get_model
from models.weights import load_released_weights, set_model_state
from utils.common import get_optimizer, get_scheduler
from utils.misc import get_logger, get_new_log_dir, inf_iterator, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, help='YAML config or a native Paddle run directory.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--weights', type=Path, help='Released weights exported to NPZ (fresh runs only).')
    parser.add_argument('--max_iters', type=int, help='Override the last training iteration.')
    parser.add_argument('--log_interval', type=int, default=1, help=
        'Log the first training update and every N updates thereafter.')
    parser.add_argument('--val_batches', type=int, help=
        'Validate on the first N batches in dataset order.')
    parser.add_argument('--record_inputs', action='store_true', help=
        'Save the inputs and diffusion noise used at each training step.')
    parser.epilog = 'Resume only from trusted native Paddle checkpoints.'
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error('--log_interval must be positive')
    if args.val_batches is not None and args.val_batches < 1:
        parser.error('--val_batches must be positive')
    if args.max_iters is not None and args.max_iters < 1:
        parser.error('--max_iters must be positive')
    resume = args.config.is_dir()
    if resume and args.weights is not None:
        parser.error('--weights cannot be used when resuming')
    if not resume and args.resume_iter is not None:
        parser.error('--resume_iter requires a native run directory')

    device = 'gpu' + args.device[4:] if args.device.startswith('cuda') else args.device
    paddle.set_device(device)
    start_iter = 1
    if resume:
        checkpoints = args.config / 'checkpoints'
        if args.resume_iter is None:
            candidates = [path for path in checkpoints.glob('*.pdckpt') if path.stem.isdecimal()]
            if not candidates:
                parser.error(f'No native Paddle checkpoints in {checkpoints}')
            ckpt_path = max(candidates, key=lambda path: int(path.stem))
        else:
            ckpt_path = checkpoints / f'{args.resume_iter}.pdckpt'
        ckpt = paddle.load(str(ckpt_path))
        config_dict = ckpt['config']
        start_iter = ckpt['iteration'] + 1
    else:
        config_dict = yaml.safe_load(args.config.read_text())
    if args.max_iters is not None:
        config_dict['train']['max_iters'] = args.max_iters
    config = EasyDict(config_dict)
    seed_all(config.train.seed)

    log_dir = Path(get_new_log_dir(args.logdir, prefix='resume' if resume else args.config.stem))
    if resume:
        (log_dir / 'resumed_from').symlink_to(args.config.resolve(), target_is_directory=True)
    else:
        shutil.copytree(Path(__file__).with_name('models'), log_dir / 'models')
    ckpt_dir = log_dir / 'checkpoints'
    ckpt_dir.mkdir()
    if args.record_inputs:
        inputs_dir = log_dir / 'inputs'
        inputs_dir.mkdir()
    logger = get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)
    config_name = 'config.yml' if resume else args.config.name
    (log_dir / config_name).write_text(yaml.safe_dump(config_dict, sort_keys=False))

    logger.info('Loading datasets...')
    train_set = GraphDataset(config.dataset.train)
    val_set = GraphDataset(config.dataset.val)
    train_iterator = inf_iterator(GraphDataLoader(train_set, batch_size=config.train.batch_size, shuffle=True))
    val_loader = GraphDataLoader(val_set, batch_size=config.train.batch_size, shuffle=False)
    val_batches = len(val_loader) if args.val_batches is None else min(args.val_batches, len(val_loader))

    logger.info('Building model...')
    model = get_model(config.model)
    clip_parameters = list({id(param): param for param in model.parameters()}.values())
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    if resume:
        logger.info('Resuming from: %s (next iteration: %d)', ckpt_path, start_iter)
        set_model_state(model, ckpt['model'])
        optimizer_global.set_state_dict(ckpt['optimizer_global'])
        optimizer_local.set_state_dict(ckpt['optimizer_local'])
        del ckpt
    elif args.weights is not None:
        logger.info('Initializing model from released weights: %s', args.weights)
        load_released_weights(model, args.weights)

    def batch_loss(batch, counts, record_inputs=None):
        return model.get_loss(
            atom_type=batch.atom_type, pos=batch.pos, bond_index=batch.edge_index,
            bond_type=batch.edge_type, batch=batch.batch, num_graphs=batch.num_graphs,
            num_nodes_per_graph=counts, anneal_power=config.train.anneal_power,
            return_unreduced_loss=True, record_inputs=record_inputs,
        )

    def train(it):
        model.train()
        optimizer_global.clear_grad(set_to_zero=False)
        optimizer_local.clear_grad(set_to_zero=False)
        batch = next(train_iterator)
        counts = paddle.diff(batch.ptr)
        record_inputs = None
        if args.record_inputs:
            def record_inputs(time_step, pos_noise):
                tensors = {
                    'atom_type': batch.atom_type, 'pos': batch.pos,
                    'bond_index': batch.edge_index, 'bond_type': batch.edge_type,
                    'batch': batch.batch, 'num_nodes_per_graph': counts,
                    'time_step': time_step, 'pos_noise': pos_noise,
                }
                np.savez(inputs_dir / f'{it}.npz', num_graphs=batch.num_graphs,
                         **{name: tensor.numpy() for name, tensor in tensors.items()})

        loss, loss_global, loss_local = batch_loss(batch, counts, record_inputs)
        loss = loss.mean()
        loss.backward()
        orig_grad_norm = paddle.nn.utils.clip_grad_norm_(clip_parameters, config.train.max_grad_norm)
        optimizer_global.step()
        optimizer_local.step()
        if (it - start_iter) % args.log_interval == 0:
            logger.info(
                '[Train] Iter %05d | Loss %.17g | Loss(Global) %.17g | Loss(Local) %.17g | Grad %.17g | LR(Global) %.17g | LR(Local) %.17g',
                it, loss.item(), loss_global.mean().item(), loss_local.mean().item(),
                orig_grad_norm.item(), optimizer_global.get_lr(), optimizer_local.get_lr(),
            )

    def validate(it):
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        with paddle.no_grad():
            model.eval()
            for batch in tqdm(islice(val_loader, val_batches), total=val_batches, desc='Validation'):
                loss, loss_global, loss_local = batch_loss(batch, paddle.diff(batch.ptr))
                sum_loss += loss.sum().item()
                sum_n += loss.shape[0]
                sum_loss_global += loss_global.sum().item()
                sum_n_global += loss_global.shape[0]
                sum_loss_local += loss_local.sum().item()
                sum_n_local += loss_local.shape[0]
        avg_loss = sum_loss / sum_n
        avg_loss_global = sum_loss_global / sum_n_global
        avg_loss_local = sum_loss_local / sum_n_local
        if config.train.scheduler.type == 'plateau':
            scheduler_global.step(avg_loss_global)
            scheduler_local.step(avg_loss_local)
        else:
            scheduler_global.step()
            scheduler_local.step()
        logger.info(
            '[Validate] Iter %05d | Loss %.17g | Loss(Global) %.17g | Loss(Local) %.17g'
            , it, avg_loss, avg_loss_global, avg_loss_local)
        return avg_loss
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validate(it)
                paddle.save({
                    'config': config_dict,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, str(ckpt_dir / f'{it}.pdckpt'))
    except KeyboardInterrupt:
        logger.info('Terminating...')


if __name__ == '__main__':
    # Native Adam matches accumulators by Paddle parameter name on the first step.
    with paddle.utils.unique_name.guard():
        main()
