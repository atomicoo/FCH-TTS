"""Wrapper trainer class for training our models."""
__author__ = 'Atomicoo'

import sys
import os
import os.path as osp
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.optim import NoamScheduler

from helpers.logger import Logger
from utils.augment import add_random_noise, degrade_some, frame_dropout
from utils.plot import plot_alignment, plot_spectrogram
from utils.utils import get_last_chkpt_path

from datasets.data_loader import Text2MelDataLoader
from datasets.dataset import SpeechDataset
from utils.transform import MinMaxNorm, StandardNorm
from utils.functional import mask

from models import DurationExtractor, ParallelText2Mel
from losses import l1_masked, guided_att, masked_huber, masked_ssim, l1_dtw
from losses.pytorch_sdtw import SoftDTW


class Trainer:
    def __init__(self,
                 model=None,
                 dataset=None,
                 compute_metrics=None,
                 optimizers=None,
                 checkpoint=None,
                 device=None
    ):
        # model, metrics, optim
        self.model = model
        self.compute_metrics = compute_metrics
        self.optimizer, self.scheduler = optimizers

        # dataset
        self.dataset = dataset

        # device
        self.device = device
        self.model.to(self.device)
        print(f'Model sent to {self.device}')

        # helper vars
        self.checkpoint = None
        self.epoch, self.step = 0, 0
        if checkpoint is not None:
            self.load_checkpoint(self.checkpoint)
    
    def to_device(self, device):
        print(f'Sending network to {device}')
        self.device = device
        self.model.to(device)
        return self
    
    def save_checkpoint(self):
        if self.checkpoint is not None:
            os.remove(self.checkpoint)
        self.checkpoint = osp.join(self.loggers.logdir, f'{time.strftime("%Y-%m-%d")}_chkpt_epoch{self.epoch:03d}.pth')
        print("Saving the checkpoint file '%s'..." % self.checkpoint)
        torch.save(
            {
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            },
            self.checkpoint)

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loaded checkpoint epoch=%d step=%d" % (self.epoch, self.step))

        self.checkpoint = None  # prevent overriding old checkpoint
        return self
    
    def train_dataloader(self, dataset, batch_size=32, \
                num_workers=0 if sys.platform.startswith('win') else 8):
        return Text2MelDataLoader(dataset, batch_size=batch_size, mode='train', num_workers=num_workers)

    def valid_dataloader(self, dataset, batch_size=32, \
                num_workers=0 if sys.platform.startswith('win') else 8):
        return Text2MelDataLoader(dataset, batch_size=batch_size, mode='valid', num_workers=num_workers)


class DurationTrainer(Trainer):
    def __init__(self,
                 hparams,
                 adam_lr=0.002,
                 warmup_epochs=30,
                 init_scale=0.25,
                 checkpoint=None,
                 device='cuda'
    ):
        self.hparams = hparams
        model = DurationExtractor(hparams.duration)
        dataset_root = osp.join(hparams.data.datasets_path, hparams.data.dataset_dir)
        dataset = SpeechDataset(['mels', 'mlens', 'texts', 'tlens'], dataset_root, hparams.text)
        compute_metrics = self.recon_losses
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
        scheduler = NoamScheduler(optimizer, warmup_epochs, init_scale)
        optimizers = (optimizer, scheduler)

        super(DurationTrainer, self).__init__(
                model=model,
                dataset=dataset,
                compute_metrics=compute_metrics,
                optimizers=optimizers,
                checkpoint=checkpoint,
                device=device
        )
    
    def fit(self, batch_size, epochs=1, chkpt_every=10, checkpoint=None, loggers=None):
        self.loggers = loggers or \
            Logger(self.hparams.trainer.logdir, self.hparams.data.dataset, 'duration')

        checkpoint = checkpoint or get_last_chkpt_path(self.loggers.logdir)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        train_loader = self.train_dataloader(copy.deepcopy(self.dataset), batch_size=batch_size)
        valid_loader = self.valid_dataloader(copy.deepcopy(self.dataset), batch_size=batch_size)

        self.normalizer = MinMaxNorm(self.hparams.audio.spec_min, self.hparams.audio.spec_max)

        for e in range(self.epoch + 1, self.epoch + 1 + epochs):
            self.epoch = e

            train_losses = self._train_epoch(train_loader)
            valid_losses = self._validate(valid_loader)

            self.scheduler.step()

            if self.epoch % chkpt_every == 0:
                # checkpoint at every 10th epoch
                self.save_checkpoint()

            self.loggers.log_epoch('train', self.epoch, 
                                   {'train_l1_loss': train_losses[1], 'train_ssim_loss': train_losses[2], 'train_att_loss': train_losses[3]})
            self.loggers.log_epoch('valid', self.epoch, 
                                   {'valid_l1_loss': valid_losses[1], 'valid_ssim_loss': valid_losses[2], 'valid_att_loss': valid_losses[3]})

            print(f'Epoch {e} | '
                    f'Train - loss: {train_losses[0]}, l1: {train_losses[1]}, ssim: {train_losses[2]}, att: {train_losses[3]}| '
                    f'Valid - loss: {valid_losses[0]}, l1: {valid_losses[1]}, ssim: {valid_losses[2]}, att: {valid_losses[3]}| ')

    def _train_epoch(self, dataloader=None):
        self.model.train()

        ll = len(dataloader)
        running_loss = 0.0
        running_l1_loss = 0.0
        running_ssim_loss = 0.0
        running_att_loss = 0.0

        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.hparams.trainer.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            self.optimizer.zero_grad()

            mels, mlens, texts, tlens = \
                batch['mels'], batch['mlens'].squeeze(1), batch['texts'].long(), batch['tlens'].squeeze(1)
            mels, mlens, texts, tlens = \
                mels.to(self.device), mlens.to(self.device), texts.to(self.device), tlens.to(self.device)

            s = mels = self.normalizer(mels)

            # Spectrogram augmentation
            if self.hparams.duration.enable_augment:
                s = add_random_noise(mels, self.hparams.duration.noise)
                s = degrade_some(self.model, s, texts, tlens, \
                                self.hparams.duration.feed_ratio, repeat=self.hparams.duration.feed_repeat)
                s = frame_dropout(s, self.hparams.duration.replace_ratio)

            melspecs, attns = self.model((texts, tlens, s, True))
            outputs_and_targets = (melspecs, mels, attns, mlens, tlens)
            loss, l1_loss, ssim_loss, att_loss = self.compute_metrics(outputs_and_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.step += 1

            loss, l1_loss, ssim_loss, att_loss = loss.item(), l1_loss.item(), ssim_loss.item(), att_loss.item()
            running_loss += loss
            running_l1_loss += l1_loss
            running_ssim_loss += ssim_loss
            running_att_loss += att_loss

            # update the progress bar
            pbar.set_postfix({
                'l1': "%.05f" % (running_l1_loss / it),
                'ssim': "%.05f" % (running_ssim_loss / it),
                'att': "%.05f" % (running_att_loss / it)
            })

            mels, melspecs, attns = mels.cpu().detach(), melspecs.cpu().detach(), attns.cpu().detach()
            index = -1
            mlen, tlen = mlens[index].item(), tlens[index].item()
            mels_fig = plot_spectrogram(melspecs[index, :mlen, :], 
                                        target_spectrogram=mels[index, :mlen, :])
            attn_fig = plot_alignment(attns[index, :mlen, :tlen])
            self.loggers.log_step('train', self.step,
                                  {'step_l1_loss': l1_loss, 'step_ssim_loss': ssim_loss, 'step_att_loss': att_loss},
                                  {'melspecs': mels_fig, 'attention': attn_fig})

        epoch_loss = running_loss / ll
        epoch_l1_loss = running_l1_loss / ll
        epoch_ssim_loss = running_ssim_loss / ll
        epoch_att_loss = running_att_loss / ll

        return epoch_loss, epoch_l1_loss, epoch_ssim_loss, epoch_att_loss

    def _validate(self, dataloader):
        self.model.eval()

        ll = len(dataloader)
        running_loss = 0.0
        running_l1_loss = 0.0
        running_ssim_loss = 0.0
        running_att_loss = 0.0

        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.hparams.trainer.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            mels, mlens, texts, tlens = \
                batch['mels'], batch['mlens'].squeeze(1), batch['texts'].long(), batch['tlens'].squeeze(1)
            mels, mlens, texts, tlens = \
                mels.to(self.device), mlens.to(self.device), texts.to(self.device), tlens.to(self.device)

            mels = self.normalizer(mels)

            with torch.no_grad():
                melspecs, attns = self.model((texts, tlens, mels, True))
            outputs_and_targets = (melspecs, mels, attns, mlens, tlens)
            loss, l1_loss, ssim_loss, att_loss = self.compute_metrics(outputs_and_targets)

            loss, l1_loss, ssim_loss, att_loss = loss.item(), l1_loss.item(), ssim_loss.item(), att_loss.item()
            running_loss += loss
            running_l1_loss += l1_loss
            running_ssim_loss += ssim_loss
            running_att_loss += att_loss

        epoch_loss = running_loss / ll
        epoch_l1_loss = running_l1_loss / ll
        epoch_ssim_loss = running_ssim_loss / ll
        epoch_att_loss = running_att_loss / ll

        return epoch_loss, epoch_l1_loss, epoch_ssim_loss, epoch_att_loss

    
    def recon_losses(self, outputs_and_targets):
        melspecs, mels, attns, mlens, tlens = outputs_and_targets
        l1_loss = l1_masked(melspecs, mels, mlens)
        ssim_loss = masked_ssim(melspecs, mels, mlens)
        att_loss = guided_att(attns, mlens, tlens)
        loss = l1_loss + ssim_loss + att_loss
        return loss, l1_loss, ssim_loss, att_loss


class ParallelTrainer(Trainer):
    def __init__(self,
                 hparams,
                 adam_lr=0.005,
                 warmup_epochs=20,
                 init_scale=0.25,
                 ground_truth=False,
                 checkpoint=None,
                 device='cuda'
    ):
        self.hparams = hparams
        model = ParallelText2Mel(hparams.parallel)
        dataset_root = osp.join(hparams.data.datasets_path, hparams.data.dataset_dir)
        dataset = SpeechDataset(['mels-gt' if ground_truth else 'mels', 'mlens', 'texts', 'tlens', 'drns'],
                                dataset_root, hparams.text)
        self.sdtw = SoftDTW(use_cuda=(device.type == 'cuda'), gamma=0.1)
        compute_metrics = self.recon_losses_v2
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        scheduler = NoamScheduler(optimizer, warmup_epochs, init_scale)
        optimizers = (optimizer, scheduler)

        super(ParallelTrainer, self).__init__(
                model=model,
                dataset=dataset,
                compute_metrics=compute_metrics,
                optimizers=optimizers,
                checkpoint=checkpoint,
                device=device
        )
    
    def fit(self, batch_size, epochs=1, chkpt_every=10, checkpoint=None, duration_file=None, loggers=None):
        self.loggers = loggers or \
            Logger(self.hparams.trainer.logdir, self.hparams.data.dataset, 'parallel')

        checkpoint = checkpoint or get_last_chkpt_path(self.loggers.logdir)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        if self.dataset.durans is None:
            self.dataset.load_durations(duration_file)
        train_loader = self.train_dataloader(copy.deepcopy(self.dataset), batch_size=batch_size)
        valid_loader = self.valid_dataloader(copy.deepcopy(self.dataset), batch_size=batch_size)

        self.normalizer = StandardNorm(self.hparams.audio.spec_mean, self.hparams.audio.spec_std)

        for e in range(self.epoch + 1, self.epoch + 1 + epochs):
            self.epoch = e
            train_losses = self._train_epoch(train_loader)
            valid_losses = self._validate(valid_loader)

            self.scheduler.step(valid_losses[0])

            if self.epoch % chkpt_every == 0:
                # checkpoint at every 10th epoch
                self.save_checkpoint()

            self.loggers.log_epoch('train', self.epoch, 
                    {'train_l1_loss': train_losses[1], 'train_ssim_loss': train_losses[2], 'train_drn_loss': train_losses[3]})
            self.loggers.log_epoch('valid', self.epoch, 
                    {'valid_l1_loss': valid_losses[1], 'valid_ssim_loss': valid_losses[2], 'valid_drn_loss': valid_losses[3]})

            print(f'Epoch {e} | '
                  f'Train - loss: {train_losses[0]}, l1: {train_losses[1]}, ssim: {train_losses[2]}, drn: {train_losses[3]}| '
                  f'Valid - loss: {valid_losses[0]}, l1: {valid_losses[1]}, ssim: {valid_losses[2]}, drn: {valid_losses[3]}| ')

    def _train_epoch(self, dataloader):
        self.model.train()

        running_loss = 0.0
        running_l1_loss = 0.0
        running_ssim_loss = 0.0
        running_drn_loss = 0.0

        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.hparams.trainer.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            self.optimizer.zero_grad()

            mels, mlens, texts, tlens, durations = \
                batch['mels'], batch['mlens'].squeeze(1), batch['texts'].long(), batch['tlens'].squeeze(1), batch['drns'].long()
            mels, mlens, texts, tlens, durations = \
                mels.to(self.device), mlens.to(self.device), texts.to(self.device), tlens.to(self.device), durations.to(self.device)

            mels = self.normalizer(mels)

            # melspecs, prd_durans = self.model((texts, tlens, durations, 1.0))
            melspecs, prd_durans = self.model((texts, tlens, None, 1.0))
            outputs_and_targets = (melspecs, mels, mlens, tlens, durations, prd_durans)
            loss, l1_loss, ssim_loss, drn_loss = self.compute_metrics(outputs_and_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.step += 1

            loss, l1_loss, ssim_loss, drn_loss = \
                loss.item(), l1_loss.item(), ssim_loss.item(), drn_loss.item()
            running_loss += loss
            running_l1_loss += l1_loss
            running_ssim_loss += ssim_loss
            running_drn_loss += drn_loss
            
            # update the progress bar
            pbar.set_postfix({
                'l1': "%.05f" % (running_l1_loss / it),
                'ssim': "%.05f" % (running_ssim_loss / it),
                'drn': "%.05f" % (running_drn_loss / it)
            })

            mels, melspecs = mels.cpu().detach(), melspecs.cpu().detach()
            index = -1
            mlen, tlen = mlens[index].item(), tlens[index].item()
            mels_fig = plot_spectrogram(melspecs[index, :mlen, :],
                                        target_spectrogram=mels[index, :mlen, :])
            self.loggers.log_step('train', self.step, 
                                  {'step_l1_loss': l1_loss, 'step_ssim_loss': ssim_loss, 'step_drn_loss': drn_loss},
                                  {'melspecs': mels_fig})

        epoch_loss = running_loss / it
        epoch_l1_loss = running_l1_loss / it
        epoch_ssim_loss = running_ssim_loss / it
        epoch_drn_loss = running_drn_loss / it

        return epoch_loss, epoch_l1_loss, epoch_ssim_loss, epoch_drn_loss

    def _validate(self, dataloader):
        self.model.eval()

        running_loss = 0.0
        running_l1_loss = 0.0
        running_ssim_loss = 0.0
        running_drn_loss = 0.0

        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.hparams.trainer.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            mels, mlens, texts, tlens, durations = \
                batch['mels'], batch['mlens'].squeeze(1), batch['texts'].long(), batch['tlens'].squeeze(1), batch['drns'].long()
            mels, mlens, texts, tlens, durations = \
                mels.to(self.device), mlens.to(self.device), texts.to(self.device), tlens.to(self.device), durations.to(self.device)

            mels = self.normalizer(mels)

            with torch.no_grad():
                melspecs, prd_durans = self.model((texts, tlens, None, 1.0))
            outputs_and_targets = (melspecs, mels, mlens, tlens, durations, prd_durans)
            loss, l1_loss, ssim_loss, drn_loss = self.compute_metrics(outputs_and_targets, use_dtw=True)

            loss, l1_loss, ssim_loss, drn_loss = \
                loss.item(), l1_loss.item(), ssim_loss.item(), drn_loss.item()
            running_loss += loss
            running_l1_loss += l1_loss
            running_ssim_loss += ssim_loss
            running_drn_loss += drn_loss

        epoch_loss = running_loss / it
        epoch_l1_loss = running_l1_loss / it
        epoch_ssim_loss = running_ssim_loss / it
        epoch_drn_loss = running_drn_loss / it

        return epoch_loss, epoch_l1_loss, epoch_ssim_loss, epoch_drn_loss
    
    def recon_losses(self, outputs_and_targets, use_dtw=False):
        melspecs, mels, mlens, tlens, durations, prd_durations  = outputs_and_targets
        if use_dtw:
            prd_mlens = prd_durations.sum(axis=-1).long()
            l1_loss = l1_dtw(mels, mlens, melspecs, prd_mlens)
            ssim_loss = torch.zeros(1)
            drn_loss = masked_huber(prd_durations, durations.float(), tlens)
            loss = l1_loss + drn_loss
        else:
            l1_loss = l1_masked(melspecs, mels, mlens)
            ssim_loss = masked_ssim(melspecs, mels, mlens)
            durations[durations < 1] = 1  # needed to prevent log(0)
            drn_loss = masked_huber(prd_durations, torch.log(durations.float()), tlens)
            loss = l1_loss + ssim_loss + drn_loss
        return loss, l1_loss, ssim_loss, drn_loss

    def recon_losses_v2(self, outputs_and_targets, use_dtw=False):
        melspecs, mels, mlens, tlens, durations, prd_durations  = outputs_and_targets
        if use_dtw:
            prd_mlens = prd_durations.sum(axis=-1).long()
            l1_loss = l1_dtw(mels, mlens, melspecs, prd_mlens)
            ssim_loss = torch.zeros(1)
            drn_loss = masked_huber(prd_durations, durations.float(), tlens)
            loss = l1_loss + drn_loss
        else:
            msk = mask(melspecs.shape, mlens, dim=1).float().to(melspecs.device)
            melspecs = melspecs * msk
            l1_loss = self.sdtw(melspecs, mels).mean()
            ssim_loss = torch.zeros(1)
            drn_loss = masked_huber(prd_durations, durations.float(), tlens)
            loss = l1_loss + drn_loss
        return loss, l1_loss, ssim_loss, drn_loss
