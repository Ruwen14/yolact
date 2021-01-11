from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import uuid
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# custom imports
import wandb

os.environ["WANDB_API_KEY"] = '394a71acf1f77ccd2c3053411559cb13b305165a'
os.environ["WANDB_MODE"] = "dryrun"
run_name = input("[INPUT] Name run:") + f'_{uuid.uuid4().hex[:4]}'
run = wandb.init(name=run_name,id=run_name, project='YOLACT')

# Oof
import eval as eval_script


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"' \
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be' \
                         'determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=1000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=3000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=True, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

wandb.config.update(args)

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]


# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))


replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out


def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))

    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

        val_data_loader = data.DataLoader(val_dataset, args.batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=True, collate_fn=detection_collate,
                                          pin_memory=True)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    wandb.watch(yolact_net)
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
                  overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print(f'Initializing weights from {args.save_folder + cfg.backbone.path}')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()

    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn()  # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)


    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types  # Forms the print order
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    wandb.config.update({
        'max_iter': cfg.max_iter,
        'epoch_size': epoch_size,
        'num_epochs': num_epochs,
        'batch_size': args.batch_size,
        'init_lr': cfg.lr,
        'init_momentum': cfg.momentum,
        'init_decay': cfg.decay,
        'init_lr_steps': cfg.lr_steps,
        'pred_aspect_ratios': [[[1, 1 / 2, 2]]] * 5,
        'pred_scales': [[24], [48], [96], [192], [384]],
    })

    best_mask_mAP = 0
    print('Configs')
    print('%-----------------------------------------------------------------------------%')
    print(f'Begin training! for {cfg.max_iter} iterations and {num_epochs} epochs at max')
    print(f'training on {len(dataset)} images | validating on {len(val_dataset)} images')
    print('preserve_aspect_ratio',cfg.preserve_aspect_ratio)
    print('LR_steps', cfg.lr_steps)
    print('Batch Size', args.batch_size)
    print('%-----------------------------------------------------------------------------%')
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            print('%----------------------------------------------------------------------------------------------%')
            print(f'     EPOCH [{epoch}|{num_epochs}]     LR [{cur_lr}]     Best Mask mAP [{best_mask_mAP} %]'   )
            print('%----------------------------------------------------------------------------------------------%')
            # Resume from start_iter
            if (epoch + 1) * epoch_size < iteration:
                continue

            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch + 1) * epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()

                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer,
                           (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                    print(f'Adjusting Learning rate to {cur_lr}')

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)

                losses = {k: (v).mean() for k, v in losses.items()}  # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])

                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
                loss.backward()  # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 20 == 0:
                    eta_str = \
                    str(datetime.timedelta(seconds=(cfg.max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[Ep:%3d] Iter:%7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                          % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log and iteration % 20 == 0:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0)  # nvidia-smi is sloooow

                    log_stuff = log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                                        lr=round(cur_lr, 10), elapsed=elapsed)

                    iter_step = log_stuff['iter']
                    wandb.log({'[TRAIN] BBox Loss': log_stuff['loss']['B']}, step=iter_step)
                    wandb.log({'[TRAIN] Mask Loss': log_stuff['loss']['M']}, step=iter_step)
                    wandb.log({'[TRAIN] Class Conf. Loss': log_stuff['loss']['C']}, step=iter_step)
                    wandb.log({'[TRAIN] Sem. Segmentation Loss': log_stuff['loss']['S']}, step=iter_step)
                    wandb.log({'[TRAIN] Overall Training Loss': log_stuff['loss']['T']}, step=iter_step)
                    wandb.log({'Learning Rate': log_stuff['lr']}, step=iter_step)
                    wandb.log({'Epoch': log_stuff['epoch']}, step=iter_step)

                    log.log_gpu_stats = args.log_gpu
                    last_iter_entry = iteration
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

            # This is done after each epoch
            if args.validation_epoch > 0:
                try:
                    cur_mask_mAP = compute_validation_map(epoch=epoch, iteration=last_iter_entry, yolact_net=yolact_net,
                                                          dataset=val_dataset,
                                                          log=log if args.log else None)
                    if cur_mask_mAP > best_mask_mAP:
                        best_mask_mAP = cur_mask_mAP
                        wandb.config.update({'best_mask_mAP':best_mask_mAP },allow_val_change=True)

                        print(f'Found new best Mask mAP with {best_mask_mAP} %, Saving weights ...\n')

                        SavePath.remove_prev_best(args.save_folder)
                        yolact_net.save_weights(save_path(epoch, repr(last_iter_entry) + f'_mAP{best_mask_mAP}'))

                    compute_validation_loss(net=net, data_loader=val_data_loader, epoch=epoch,
                                            iteration=last_iter_entry,
                                            log=log if args.log else None)
                except KeyboardInterrupt:
                    if args.interrupt:
                        print('Stopping early. Saving network...')

                        # Delete previous copy of the interrupted network so we don't spam the weights folder
                        SavePath.remove_interrupt(args.save_folder)

                        yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
                        run.save()
                    sys.exit()

        # Compute validation mAP after training is finished
        compute_validation_map(epoch=epoch, iteration=last_iter_entry, yolact_net=yolact_net,dataset=val_dataset,log=log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')

            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)

            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
            run.save()
        sys.exit()
    yolact_net.save_weights(save_path(epoch, iteration))
    run.save()


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    global cur_lr
    cur_lr = new_lr


def gradinator(x):
    x.requires_grad = False
    return x


def prepare_data(datum, devices: list = None, allocation: list = None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation))  # The rest might need more/less

        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx] = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx] = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images) - 1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx] = torch.stack(images[cur_idx:cur_idx + alloc], dim=0)
            split_targets[device_idx] = targets[cur_idx:cur_idx + alloc]
            split_masks[device_idx] = masks[cur_idx:cur_idx + alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx + alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds


def no_inf_mean(x: torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()


def compute_validation_loss(net, data_loader, epoch, iteration,  log: Log = None):
    global loss_types

    with torch.no_grad():
        print(f'Computing validation Loss for {args.validation_size} images (this may take a while)...', flush=True)
        start = time.time()
        losses = {}
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:

            _losses = net(datum)

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break

        for k in losses:
            losses[k] /= iterations

        total_loss = sum([losses[k] for k in losses])
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print()
        print(('|| Validation ||' + (' %s: %.3f |' * len(losses)) + f' T:{total_loss:.3f} |') % tuple(loss_labels), flush=True)
        print()
        precision = 5
        loss_info = {k: round(losses[k], precision) for k in losses}
        loss_info['T'] = round(total_loss, precision)

        end = time.time()

        log_stuffee=log.log('val', loss=loss_info, epoch=epoch, iter=iteration, elapsed=(end-start))

        val_iter_step_loss = log_stuffee['iter']
        wandb.log({'[VAL] BBox Loss': log_stuffee['loss']['B']}, step=val_iter_step_loss)
        wandb.log({'[VAL] Mask Loss': log_stuffee['loss']['M']}, step=val_iter_step_loss)
        wandb.log({'[VAL] Class Conf. Loss': log_stuffee['loss']['C']}, step=val_iter_step_loss)
        wandb.log({'[VAL] Sem. Segmentation Loss': log_stuffee['loss']['S']}, step=val_iter_step_loss)
        wandb.log({'[VAL] Overall Validation Loss': log_stuffee['loss']['T']}, step=val_iter_step_loss)
        # wandb.log({'Epoch Val': log_stuffee['epoch']})



def compute_validation_map(epoch, iteration, yolact_net, dataset, log: Log = None):
    with torch.no_grad():
        yolact_net.eval()

        start = time.time()
        print()
        print(f'Computing validation mAP for {args.validation_size} images (this may take a while)...', flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log_stuffe = log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)
            val_iter_step = log_stuffe['iter']
            wandb.log({'[VAL] Averaged BBox mAP': log_stuffe['box']['all']}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 50%': log_stuffe['box'][50]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 55%': log_stuffe['box'][55]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 60%': log_stuffe['box'][60]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 65%': log_stuffe['box'][65]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 70%': log_stuffe['box'][70]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 75%': log_stuffe['box'][75]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 80%': log_stuffe['box'][80]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 85%': log_stuffe['box'][85]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 90%': log_stuffe['box'][90]}, step=val_iter_step)
            wandb.log({'[VAL] BBox mAP @ 95%': log_stuffe['box'][95]}, step=val_iter_step)

            wandb.log({'[VAL] Averaged Mask mAP': log_stuffe['mask']['all']}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 50%': log_stuffe['mask'][50]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 55%': log_stuffe['mask'][55]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 60%': log_stuffe['mask'][60]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 65%': log_stuffe['mask'][65]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 70%': log_stuffe['mask'][70]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 75%': log_stuffe['mask'][75]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 80%': log_stuffe['mask'][80]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 85%': log_stuffe['mask'][85]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 90%': log_stuffe['mask'][90]}, step=val_iter_step)
            wandb.log({'[VAL] Mask mAP @ 95%': log_stuffe['mask'][95]}, step=val_iter_step)
            # wandb.log({'Epoch Val': log_stuffe['epoch']})

        yolact_net.train()
        return log_stuffe['mask']['all']


def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images=' + str(args.validation_size)])


if __name__ == '__main__':
    train()

# python train.py --config=yolact_resnet50_custom_car_config  --save_interval=1000 --validation_size=3000
#python train_custom.py --config=yolact_resnet50_custom_car_config  --save_interval=1000 --validation_size=1328 --num_workers=4
#python train_custom.py --config=yolact_resnet101_custom_car_config  --save_interval=1000 --validation_size=1328 --num_workers=4