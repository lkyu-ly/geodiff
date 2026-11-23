import paddle
import copy
import warnings
import numpy as np
from torch_geometric.data import Data, Batch


class ExponentialLR_with_minLr(paddle.optimizer.lr.ExponentialDecay):

    def __init__(self, optimizer, gamma, min_lr=0.0001, last_epoch=-1,
        verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma,
            last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                'To get the last learning rate computed by the scheduler, please use `get_last_lr()`.'
                , UserWarning)
        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr) for group in
            self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr) for
            base_lr in self.base_lrs]


def repeat_data(data: Data, num_repeat) ->Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) ->Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return paddle.optimizer.Adam(parameters=model.parameters(),
            learning_rate=cfg.lr, weight_decay=cfg.weight_decay, beta1=(cfg
            .beta1, cfg.beta2)[0], beta2=(cfg.beta1, cfg.beta2)[1])
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(factor=cfg.factor,
            patience=cfg.patience, learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        return tmp_lr
    elif cfg.type == 'expmin':
        return ExponentialLR_with_minLr(optimizer, gamma=cfg.factor, min_lr
            =cfg.min_lr)
    elif cfg.type == 'expmin_milestone':
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(optimizer, gamma=gamma, min_lr=cfg.
            min_lr)
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)
