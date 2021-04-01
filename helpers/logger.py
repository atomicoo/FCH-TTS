"""Wrapper class for logging into the TensorBoard and comet.ml"""
__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['Logger']

import os
from tensorboardX import SummaryWriter
try:
    import wandb
except ImportError:
    wandb = None


class Logger(object):

    def __init__(self, logdir, dataset_name, model_name, wandb_info=None):
        self.model_name = model_name
        self.project_name = "%s-%s" % (dataset_name, self.model_name)
        self.logdir = os.path.join(logdir, self.project_name)
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.wandb = None if wandb_info is None else wandb
        if self.wandb and self.wandb.run is None:
            self.wandb.init(**wandb_info)

    def log_model(self, model):
        self.writer.add_graph(model)
        if self.wandb is not None:
            self.wandb.watch(model)

    def log_step(self, phase, step, loss_dict, image_dict):
        if phase == 'train':
            if step % 2 == 0:
                # self.writer.add_scalar('lr', get_lr(), step)
                # self.writer.add_scalar('%s-step/loss' % phase, loss, step)
                for key in sorted(loss_dict):
                    self.writer.add_scalar(f"{phase}-step/{key}", loss_dict[key], step)
                if self.wandb is not None:
                    self.wandb.log(loss_dict)

            if step % 10 == 0:
                for key in sorted(image_dict):
                    self.writer.add_figure(f"{self.model_name}/{key}", image_dict[key], step)
                if self.wandb is not None:
                    self.wandb.log({k: self.wandb.Image(v) for k,v in image_dict.items()})

    def log_epoch(self, phase, epoch, loss_dict):
        for key in sorted(loss_dict):
            self.writer.add_scalar(f"{phase}/{key}", loss_dict[key], epoch)
        if self.wandb is not None:
            self.wandb.log(loss_dict)
