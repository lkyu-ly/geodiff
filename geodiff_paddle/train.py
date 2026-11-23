import paddle
import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
from itertools import islice
import numpy as np
from torch_geometric.data import DataLoader
from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
from utils.common import get_optimizer, get_scheduler
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--log_interval', type=int, default=1, help=
        'Log the first training update and every N updates thereafter.')
    parser.add_argument('--val_batches', type=int, help=
        'Validate on the first N batches in dataset order.')
    parser.add_argument('--record_inputs', action='store_true', help=
        'Save the inputs and diffusion noise used at each training step.')
    parser.add_argument('--trust_checkpoint', action='store_true', help=
        'Allow pickle loading for a checkpoint from a trusted source.')
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error('--log_interval must be positive')
    if args.val_batches is not None and args.val_batches < 1:
        parser.error('--val_batches must be positive')
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(
        config_path).rfind('.')]
    seed_all(config.train.seed)
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume'
            )
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.
            path.basename(resume_from.rstrip('/'))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.record_inputs:
        inputs_dir = os.path.join(log_dir, 'inputs')
        os.makedirs(inputs_dir)
    logger = get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(
        config_path)))
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    val_set = ConformationDataset(config.dataset.val, transform=transforms)
    train_iterator = inf_iterator(DataLoader(train_set, config.train.
        batch_size, shuffle=True))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)
    val_batches = len(val_loader) if args.val_batches is None else min(args
        .val_batches, len(val_loader))
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global
        )
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    start_iter = 1
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(
            resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        logger.info('Iteration: %d' % start_iter)
        ckpt = paddle.load(path=str(ckpt_path))
        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])

    def train(it):
        model.train()
        optimizer_global.zero_grad()
        optimizer_local.zero_grad()
        batch = next(train_iterator).to(args.device)
        record_inputs = None
        if args.record_inputs:

            def record_inputs(time_step, pos_noise):
                tensors = {'atom_type': batch.atom_type, 'pos': batch.pos,
                    'bond_index': batch.edge_index, 'bond_type': batch.
                    edge_type, 'batch': batch.batch, 'num_nodes_per_graph':
                    batch.num_nodes_per_graph, 'time_step': time_step,
                    'pos_noise': pos_noise}
                np.savez(os.path.join(inputs_dir, f'{it}.npz'), num_graphs=
                    batch.num_graphs, **{name: tensor.detach().cpu().numpy(
                    ) for name, tensor in tensors.items()})
        loss, loss_global, loss_local = model.get_loss(atom_type=batch.
            atom_type, pos=batch.pos, bond_index=batch.edge_index,
            bond_type=batch.edge_type, batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph, num_graphs=batch
            .num_graphs, anneal_power=config.train.anneal_power,
            return_unreduced_loss=True, record_inputs=record_inputs)
        loss = loss.mean()
        loss.backward()
        orig_grad_norm = paddle.nn.utils.clip_grad_norm_(parameters=model.
            parameters(), max_norm=config.train.max_grad_norm)
        optimizer_global.step()
        optimizer_local.step()
        if (it - start_iter) % args.log_interval == 0:
            logger.info(
                '[Train] Iter %05d | Loss %.17g | Loss(Global) %.17g | Loss(Local) %.17g | Grad %.17g | LR(Global) %.17g | LR(Local) %.17g'
                , it, loss.item(), loss_global.mean().item(), loss_local.
                mean().item(), orig_grad_norm.item(), optimizer_global.
                param_groups[0]['lr'], optimizer_local.param_groups[0]['lr'])

    def validate(it):
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        with paddle.no_grad():
            model.eval()
            for batch in tqdm(islice(val_loader, val_batches), total=
                val_batches, desc='Validation'):
                batch = batch.to(args.device)
                loss, loss_global, loss_local = model.get_loss(atom_type=
                    batch.atom_type, pos=batch.pos, bond_index=batch.
                    edge_index, bond_type=batch.edge_type, batch=batch.
                    batch, num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs, anneal_power=config.train.
                    anneal_power, return_unreduced_loss=True)
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
                sum_loss_global += loss_global.sum().item()
                sum_n_global += loss_global.size(0)
                sum_loss_local += loss_local.sum().item()
                sum_n_local += loss_local.size(0)
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
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                paddle.save(obj={'config': config, 'model': model.
                    state_dict(), 'optimizer_global': optimizer_global.
                    state_dict(), 'scheduler_global': scheduler_global.
                    state_dict(), 'optimizer_local': optimizer_local.
                    state_dict(), 'scheduler_local': scheduler_local.
                    state_dict(), 'iteration': it, 'avg_val_loss':
                    avg_val_loss}, path=ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
