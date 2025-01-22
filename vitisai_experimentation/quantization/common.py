import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

quantizer_norm=True
train_batch_size=64
weight_lr=1e-5
weight_decay=1e-4
quantizer_lr=1e-2
epochs=3
display_freq=100
val_freq=1000
save_dir='./qat_models'
weight_lr_decay=0.94
quantizer_lr_decay=0.5
def train_one_step(model, inputs, criterion, optimizer, step, device):
  # switch to train mode
  model.train()

  images, target = inputs
  if not isinstance(model, nn.DataParallel):
    model = model.to(device)
  images = images.to(device, non_blocking=True)
  target = target.to(device, non_blocking=True)

  # compute output
  output = model(images)
  loss = criterion(output, target)

  l2_decay = 1e-4
  l2_norm = 0.0
  q_params = model.quantizer_parameters() if not isinstance(model, nn.DataParallel) \
    else model.module.quantizer_parameters()
  for param in q_params:
    l2_norm += torch.pow(param, 2.0)[0]
  if quantizer_norm:
    loss += l2_decay * torch.sqrt(l2_norm)

  # measure accuracy and record loss
  acc1, acc5 = accuracy(output, target, topk=(1, 5))

  # compute gradient and do SGD step
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss, acc1, acc5

def validate(val_loader, model, criterion, device):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

  # switch to evaluate mode
  model.eval()
  if not isinstance(model, nn.DataParallel):
    model = model.to(device)

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
      images = images.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)

      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return top1.avg

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):

  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, train_loader, val_loader, criterion, device_ids):
  best_acc1 = 0
  best_filepath = None

  num_train_batches_per_epoch = int(len(train_loader) / train_batch_size)
  '''if device_ids is not None and len(device_ids) > 0:
    device = f"cuda:{device_ids[0]}"
    model = model.to(device)
    if len(device_ids) > 1:
      model = nn.DataParallel(model, device_ids=device_ids)'''
  device='cpu'

  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')

  param_groups = [{
      'params': model.quantizer_parameters() if not isinstance(model, nn.DataParallel) \
        else model.module.quantizer_parameters(),
      'lr': quantizer_lr,
      'name': 'quantizer'
  }, {
      'params': model.non_quantizer_parameters() if not isinstance(model, nn.DataParallel) \
        else model.module.non_quantizer_parameters(),
      'lr': weight_lr,
      'name': 'weight'
  }]
  optimizer = torch.optim.Adam(
      param_groups, weight_lr, weight_decay=weight_decay)

  for epoch in range(epochs):
    progress = ProgressMeter(
        len(train_loader) * epochs,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch[{}], Step: ".format(epoch))

    for i, (images, target) in enumerate(train_loader):
      end = time.time()
      # measure data loading time
      data_time.update(time.time() - end)

      step = len(train_loader) * epoch + i

      adjust_learning_rate(optimizer, epoch, step)
      loss, acc1, acc5 = train_one_step(model, (images, target), criterion,
                                        optimizer, step, device)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      if step % display_freq == 0:
        progress.display(step)

      if step % val_freq == 0:
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        filepath = save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) \
                  else model.module.state_dict(),
                'best_acc1': best_acc1
            }, is_best, save_dir)
        if is_best:
          best_filepath = filepath

  return best_filepath

def adjust_learning_rate(optimizer, epoch, step):
  """Sets the learning rate to the initial LR decayed by decay ratios"""

  weight_lr_decay_steps = 3000 * (24 / train_batch_size)
  quantizer_lr_decay_steps = 1000 * (24 / train_batch_size)

  for param_group in optimizer.param_groups:
    group_name = param_group['name']
    if group_name == 'weight' and step % weight_lr_decay_steps == 0:
      lr = weight_lr * (
          weight_lr_decay**(step / weight_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))
    if group_name == 'quantizer' and step % quantizer_lr_decay_steps == 0:
      lr = quantizer_lr * (
          quantizer_lr_decay**(step / quantizer_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))

def save_checkpoint(state, is_best, directory):
  mkdir_if_not_exist(directory)

  filepath = os.path.join(directory, 'model.pth')
  torch.save(state, filepath)
  if is_best:
    best_acc1 = state['best_acc1'].item()
    best_filepath = os.path.join(directory, 'model_best_%5.3f.pth' % best_acc1)
    shutil.copyfile(filepath, best_filepath)
    print('Saving best ckpt to {}, acc1: {}'.format(best_filepath, best_acc1))
  return best_filepath if is_best else filepath

def mkdir_if_not_exist(x):
  if not x or os.path.isdir(x):
    return
  os.mkdir(x)
  if not os.path.isdir(x):
    raise RuntimeError("Failed to create dir %r" % x)


