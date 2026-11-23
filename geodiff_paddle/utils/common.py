import copy

import paddle

from graph_data import Batch, Data


class ExponentialLR_with_minLr(paddle.optimizer.lr.LRScheduler):

    def __init__(self, learning_rate, gamma, min_lr=0.0001):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(learning_rate)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        return max(self.base_lr * self.gamma ** self.last_epoch, self.min_lr)


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
        return paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay, beta1=cfg.beta1, beta2=cfg.beta2,
        )
    raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    learning_rate = optimizer.get_lr()
    if cfg.type == 'plateau':
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            factor=cfg.factor, patience=cfg.patience, learning_rate=learning_rate,
        )
    elif cfg.type in ('expmin', 'expmin_milestone'):
        gamma = cfg.factor if cfg.type == 'expmin' else cfg.factor ** (1 / cfg.milestone)
        scheduler = ExponentialLR_with_minLr(learning_rate, gamma=gamma, min_lr=cfg.min_lr)
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)
    optimizer.set_lr_scheduler(scheduler)
    return scheduler
