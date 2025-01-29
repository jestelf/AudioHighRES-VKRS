# ----------------------------------------------------------------------------
# train.py (адаптированный вариант с использованием встроенного AMP от PyTorch)
# ----------------------------------------------------------------------------
import os
import time
import argparse
import math
from numpy import finfo

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from distributed import apply_gradient_allreduce
from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(
        backend=hparams.dist_backend,
        init_method=hparams.dist_url,
        world_size=n_gpus,
        rank=rank,
        group_name=group_name
    )
    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Загружаем датасеты (train + val)
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)

    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn
    )
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        tmp = model.state_dict()
        tmp.update(model_dict)
        model_dict = tmp
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({
        'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate
    }, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(
            valset,
            sampler=val_sampler,
            num_workers=1,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=collate_fn
        )

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            # Без amp, т.к. eval
            y_pred = model(x)
            loss = criterion(y_pred, y)

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss

        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print(f"Validation loss {iteration}: {val_loss:.6f}")
        if logger is not None:
            logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start,
          n_gpus, rank, group_name, hparams):
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=hparams.weight_decay
    )

    # Инициализируем GradScaler без параметра device
    scaler = GradScaler(enabled=hparams.fp16_run, device='cuda')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()
    logger = prepare_directories_and_logger(output_directory, log_directory, rank)
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    iteration = 0
    epoch_offset = 0

    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _lr, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _lr
            iteration += 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    for epoch in range(epoch_offset, hparams.epochs):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)

            with autocast(device_type='cuda', enabled=hparams.fp16_run):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh
            )
            scaler.step(optimizer)
            scaler.update()

            duration = time.perf_counter() - start
            if rank == 0:
                print(f"Train loss {iteration} {reduced_loss:.6f} GradNorm {grad_norm:.6f} {duration:.2f}s/it")
                if logger is not None:
                    logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

            if (iteration % hparams.iters_per_checkpoint == 0):
                validate(
                    model, criterion, valset, iteration,
                    hparams.batch_size, n_gpus, collate_fn, logger,
                    hparams.distributed_run, rank
                )
                if rank == 0:
                    ckpt_path = os.path.join(output_directory, f"checkpoint_{iteration}")
                    save_checkpoint(model, optimizer, learning_rate, iteration, ckpt_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, required=True,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling, "(теперь не используется — встроенный amp)")
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory,
          args.checkpoint_path,
          args.warm_start,
          args.n_gpus,
          args.rank,
          args.group_name,
          hparams)
