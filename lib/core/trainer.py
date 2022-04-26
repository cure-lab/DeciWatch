import torch
import logging

from lib.core.loss import *
from progress.bar import Bar

from lib.utils.eval_metrics import *
from lib.utils.geometry_utils import *
from lib.models.smpl import SMPL

import time
import os
import shutil

logger = logging.getLogger(__name__)


class Trainer():  # merge

    def __init__(self,
                 cfg,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss,
                 writer,
                 optimizer,
                 start_epoch=0):
        super().__init__()
        self.cfg = cfg

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        # Training parameters
        self.writer = writer
        self.logdir = cfg.LOGDIR

        self.start_epoch = start_epoch
        self.end_epoch = cfg.TRAIN.EPOCH
        self.epoch = 0

        self.train_global_step = 0
        self.valid_global_step = 0
        self.device = cfg.DEVICE
        self.resume = cfg.TRAIN.RESUME
        self.lr = cfg.TRAIN.LR

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        # Resume from a pretrained model
        if self.resume is not None:
            self.resume_pretrained(self.resume)

    def run(self):
        logger.info("\n")
        for epoch_num in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch_num
            self.train()
            if self.cfg.TRAIN.VALIDATE:
                performance = self.evaluate()
                self.save_model(performance, epoch_num)
            else:
                self.save_model(None, epoch_num)

            # Decay learning rate exponentially
            lr_decay = self.cfg.TRAIN.LRDECAY
            self.lr *= lr_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

            logger.info("\n")
        self.writer.close()
        if not self.cfg.TRAIN.VALIDATE:
            performance = self.evaluate()

    def train(self):

        self.model.train()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        start = time.time()
        summary_string = ''
        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}',
                  fill='*',
                  max=len(self.train_dataloader))

        for i, data in enumerate(self.train_dataloader):

            data_pred = data["pred"].to(self.device)
            data_gt = data["gt"].to(self.device)

            timer['data'] = time.time() - start
            start = time.time()

            self.optimizer.zero_grad()

            predicted_3d_pos, denoised_3d_pos = self.model(
                data_pred, self.device)

            timer['forward'] = time.time() - start
            start = time.time()

            loss_total = self.loss(predicted_3d_pos, denoised_3d_pos, data_gt,
                                   self.model.encoder_mask,
                                   self.model.decoder_mask,
                                   self.cfg.TRAIN.USE_SMPL_LOSS)

            timer['loss'] = time.time() - start
            start = time.time()

            loss_total.backward()
            self.optimizer.step()

            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer[
                'loss'] + timer['backward']

            summary_string = f'(Iter {i + 1} | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {loss_total:.4f}'

            self.writer.add_scalar('train_loss',
                                   loss_total,
                                   global_step=self.train_global_step)

            for k, v in timer.items():
                summary_string += f' | time_{k}: {v:.2f}'

            summary_string += f' | learning rate: {self.lr}'

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(loss_total):
                exit('Nan value in loss, exiting!...')

        logger.info(summary_string)

    def evaluate_3d(self):

        eval_dict = evaluate_deciwatch_3D(self.model, self.test_dataloader,
                                          self.device, self.cfg)

        log_str = f'Epoch {self.epoch+1}, '
        log_str += ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        logger.info(log_str)

        for k, v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return eval_dict

    def evaluate_smpl(self):
        eval_dict = evaluate_deciwatch_smpl(self.model, self.test_dataloader,
                                            self.device, self.cfg)

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        logger.info(log_str)

        for k, v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return eval_dict

    def evaluate_2d(self):

        eval_dict = evaluate_deciwatch_2D(self.model, self.test_dataloader,
                                          self.device, self.cfg)

        log_str = f'Epoch {self.epoch+1}, '
        log_str += ' '.join(
            [f'{k.upper()}: {v*100:.2f}%,' for k, v in eval_dict.items()])
        logger.info(log_str)

        for k, v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return eval_dict

    def evaluate(self):

        self.model.eval()
        if self.cfg.BODY_REPRESENTATION == "3D":
            return self.evaluate_3d()

        elif self.cfg.BODY_REPRESENTATION == "smpl":
            return self.evaluate_smpl()

        elif self.cfg.BODY_REPRESENTATION == "2D":
            return self.evaluate_2d()

    def resume_pretrained(self, model_path):
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.performance = checkpoint['performance']

            logger.info(
                f"=> loaded checkpoint '{model_path}' "
                f"(epoch {self.start_epoch}, performance {self.performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict()
        }

        filename = os.path.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)
