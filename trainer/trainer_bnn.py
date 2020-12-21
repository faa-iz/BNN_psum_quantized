import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
from utils import AverageMeter
from utils import filt_alpha_weight, get_reg, get_reg_lr, accuracy, adjust_optimizer
import time
from datetime import datetime
from pytorch_memlab import MemReporter


class TrainerBNN(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config['trainer']['print_freq']

        if self.lr_scheduler == None:
            self.regime = getattr(self.model, 'regime', {0: {'optimizer': 'Adam',
                                           'lr': 1e-3,
                                           'weight_decay': 1e-4}})
            self.optimizer = adjust_optimizer(self.optimizer, 0, self.regime)


        # init variables
        self.dummy_forward()
        self.loss = torch.nn.CrossEntropyLoss()

    def dummy_forward(self):
        for i, (inputs, target) in enumerate(self.data_loader):

            input_var = inputs.to(self.device)
            output = self.model(input_var)
            break
        del output
        if 'cuda' in self.device.type:
            torch.cuda.empty_cache()
        else:
            torch.empty_cache()
        return

    def update_params(self, alpha, weights, config, clip_scale=2.0):
        if not len(alpha) > 0 and not config['no_clip']:
            for p in list(self.model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            self.optimizer.step()
            for p in list(self.model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))
        elif not config['no_clip']:
            for w in weights:
                w.data.copy_(w.org)
            self.optimizer.step()
            for a, w in zip(alpha, weights):
                with torch.no_grad():
                    w.org.copy_(torch.where(w > clip_scale*a, clip_scale*a, torch.where(w < -clip_scale*a, -clip_scale*a, w)))
        elif config['no_clip']:
            for p in list(self.model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            self.optimizer.step()
            for p in list(self.model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data)



    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        # restore weights prior to training, as we add regularization...
        for p in list(self.model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            try:
                self.writer.add_histogram(name, p, bins='auto')
            except ValueError:
                pass



        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        reg_loss = AverageMeter()
        weight_freeze = False

        alpha, weights = filt_alpha_weight(self.model)
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # measure data loading time
            data_time.update(time.time() - end)


            data, target = data.to(self.device), target.to(self.device)

            reg = get_reg(alpha, weights, self.config['model']['args'].get('reg_type', ''))
            reg *= get_reg_lr(epoch, self.config['model']['args'].get('reg_lr', ''))



            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss(output, target)

            loss.backward()
            if (self.config['model']['args'].get('full_precision', False)):
                self.optimizer.step()
            else:
                self.update_params(alpha, weights, self.config['model']['args'], self.config['model']['args'].get('clip_scale', 2.0))

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            losses.update(loss.item(), data.size(0))
            if (type(reg) == torch.Tensor):
                reg_loss.update(reg.item(), data.size(0))
            else:
                reg_loss.update(reg, data.size(0))


            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)
            batch_time.update(time.time() - end)
            end = time.time()



            if batch_idx % self.log_step == 0:
                #self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                #    epoch,
                #    self._progress(batch_idx),
                #    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                self.logger.debug('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Reg Loss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, batch_idx, len(self.data_loader),
                             phase='TRAINING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, reg_loss=reg_loss, top1=top1, top5=top5))

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(eval(self.config['lr_scheduler'].get('metric', "None")))
        else:
            self.optimizer = adjust_optimizer(self.optimizer, epoch, self.regime)


        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))


        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
