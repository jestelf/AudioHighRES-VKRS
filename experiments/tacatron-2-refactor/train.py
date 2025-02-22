# -*- coding: utf-8 -*-

"""
train.py
Адаптированный вариант скрипта тренировки модели Tacotron2,
использующий встроенный AMP (автоматическую смешанную точность) от PyTorch.
Поддерживает запуск в распределённом режиме (DistributedDataParallel).
"""

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
    """
    Суммирует тензоры со всех GPU и нормирует результат.
    Используется при распределённом обучении, чтобы получить
    среднее значение лосса или градиентов по всем устройствам.

    Параметры:
    ----------
    tensor : torch.Tensor
        Тензор, который нужно "редьюснуть" (собрать) со всех GPU.
    n_gpus : int
        Общее количество GPU.

    Возвращает:
    -----------
    rt : torch.Tensor
        Усреднённый тензор (после распределённой all_reduce).
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    """
    Инициализация процесса для распределённого обучения (DDP).
    Устанавливает текущий девайс и запускает init_process_group.

    Параметры:
    ----------
    hparams : object
        Гиперпараметры, содержащие dist_backend и dist_url.
    n_gpus : int
        Общее число GPU в системе/кластере.
    rank : int
        Текущий ранг (номер) процесса/GPU.
    group_name : str
        Имя распределённой группы процессов.
    """
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
    """
    Создаёт объекты DataLoader для тренировочного и валидационного датасета.
    Использует TextMelLoader для загрузки данных и TextMelCollate для
    подготовки батчей.

    Параметры:
    ----------
    hparams : object
        Гиперпараметры, в частности пути к training_files и validation_files,
        а также n_frames_per_step, batch_size и др.

    Возвращает:
    -----------
    train_loader : DataLoader
        DataLoader для тренировочного датасета.
    valset : TextMelLoader
        Валидационный датасет (используется при валидации).
    collate_fn : TextMelCollate
        Функция коллации (склейки батча).
    """
    # Создаём объекты для тренировочного и валидационного датасета
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)

    # Функция склейки батчей
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    # Если распределённый режим, то делаем DistributedSampler
    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # DataLoader для тренировочного датасета
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
    """
    Создаёт директорию для чекпоинтов и инициализирует объект для логгирования.

    Параметры:
    ----------
    output_directory : str
        Путь к директории, где будут сохраняться чекпоинты.
    log_directory : str
        Путь к директории, где будут сохраняться логи (TensorBoard).
    rank : int
        Ранк (номер) процесса. Логгер создаём только на rank=0,
        чтобы не плодить дубликаты логов.

    Возвращает:
    -----------
    logger : Tacotron2Logger или None
        Логгер для записи метрик и аудио в TensorBoard (если rank==0),
        иначе None.
    """
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    """
    Создаёт и возвращает экземпляр модели Tacotron2, перемещая на GPU,
    а также проводит настройки под FP16 при необходимости.

    Параметры:
    ----------
    hparams : object
        Гиперпараметры, необходимые для инициализации модели.

    Возвращает:
    -----------
    model : nn.Module
        Экземпляр модели Tacotron2 на GPU.
    """
    # Создаём модель
    model = Tacotron2(hparams).cuda()

    # Если режим fp16, меняем маску в attention_layer
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    # Если распределённый режим, оборачиваем модель в apply_gradient_allreduce
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    """
    "Тёплый" старт модели: загружает веса из чекпоинта, но позволяет
    игнорировать некоторые слои (например, если количество спикеров изменилось).

    Параметры:
    ----------
    checkpoint_path : str
        Путь к файлу чекпоинта.
    model : nn.Module
        Экземпляр модели, в которую загружаем веса.
    ignore_layers : list of str
        Список имён слоёв, которые нужно пропустить при загрузке.

    Возвращает:
    -----------
    model : nn.Module
        Модель с частично загруженными весами.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']

    if len(ignore_layers) > 0:
        # Убираем игнорируемые слои из state_dict
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        tmp = model.state_dict()
        tmp.update(model_dict)
        model_dict = tmp

    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Полная загрузка модели и оптимизатора из чекпоинта (используется при дообучении).

    Параметры:
    ----------
    checkpoint_path : str
        Путь к файлу чекпоинта.
    model : nn.Module
        Модель Tacotron2.
    optimizer : torch.optim.Optimizer
        Optimizer, который обучает модель.

    Возвращает:
    -----------
    model : nn.Module
        Модель с загруженными весами.
    optimizer : torch.optim.Optimizer
        Optimizer с загруженным состоянием.
    learning_rate : float
        Текущее значение learning_rate, сохранённое в чекпоинте.
    iteration : int
        Последняя итерация обучения, при которой был сохранён чекпоинт.
    """
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
    """
    Сохраняет текущее состояние модели и оптимизатора в файл.

    Параметры:
    ----------
    model : nn.Module
        Модель Tacotron2.
    optimizer : torch.optim.Optimizer
        Optimizer, обучающий модель.
    learning_rate : float
        Текущее значение learning rate.
    iteration : int
        Текущая итерация обучения (номер).
    filepath : str
        Куда сохранить чекпоинт.
    """
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({
        'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate
    }, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """
    Проводит валидацию модели на отложенном датасете (valset), вычисляя средний лосс.
    Выводит (и при необходимости логгирует) результат.

    Параметры:
    ----------
    model : nn.Module
        Модель Tacotron2.
    criterion : Tacotron2Loss
        Функция потерь, которая рассчитывает лосс для (y_pred, y).
    valset : TextMelLoader
        Объект датасета для валидации.
    iteration : int
        Текущая итерация (используется для логов).
    batch_size : int
        Размер батча.
    n_gpus : int
        Количество GPU, если распределённый режим.
    collate_fn : callable
        Функция, которая склеивает (collate) батчи из valset.
    logger : Tacotron2Logger или None
        Логгер для записи результатов (только на rank=0).
    distributed_run : bool
        Флаг, запускается ли обучение в распределённом режиме.
    rank : int
        Текущий ранг процесса.
    """
    model.eval()
    with torch.no_grad():
        # Создаём Dataloader для валидации
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
            # Прогоняем батч через модель
            x, y = model.parse_batch(batch)
            # В режиме eval не используем amp (нет backward, нет выгоды)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # Если распределённый режим, усредняем лосс
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()

            val_loss += reduced_val_loss

        # Средний лосс по всему датасету
        val_loss = val_loss / (i + 1)

    model.train()

    # Логгируем и печатаем только на rank=0
    if rank == 0:
        print(f"Validation loss {iteration}: {val_loss:.6f}")
        if logger is not None:
            logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start,
          n_gpus, rank, group_name, hparams):
    """
    Основная функция тренировки модели Tacotron2.
    1) Инициализирует распределённое обучение (если нужно).
    2) Создаёт и загружает модель, оптимизатор, шедулер и т.д.
    3) Запускает основной цикл обучения по эпохам и батчам,
       периодически делая валидацию и сохраняя чекпоинты.

    Параметры:
    ----------
    output_directory : str
        Путь к директории, куда сохранять чекпоинты.
    log_directory : str
        Путь к директории, где будут созданы логи (TensorBoard).
    checkpoint_path : str
        Если не None, путь к файлу чекпоинта, откуда продолжить обучение.
    warm_start : bool
        Если True, то игнорирует некоторые слои при загрузке (hparams.ignore_layers).
    n_gpus : int
        Общее число GPU для распределённого обучения.
    rank : int
        Ранк (номер) текущего процесса/GPU.
    group_name : str
        Имя группы процессов для DDP.
    hparams : объект
        Гиперпараметры модели (см. hparams.py).
    """
    # Инициализация DDP (если нужно)
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    # Фиксируем seed для воспроизводимости
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # Загружаем модель (на GPU)
    model = load_model(hparams)
    learning_rate = hparams.learning_rate

    # Создаём оптимизатор (Adam)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=hparams.weight_decay
    )

    # Инициализируем GradScaler для AMP
    scaler = GradScaler(enabled=hparams.fp16_run, device='cuda')

    # Если распределённый режим, ещё раз применяем wrap (для совместимости)
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    # Функция потерь
    criterion = Tacotron2Loss()

    # Готовим директории для чекпоинтов и логгер
    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    # Создаём DataLoader'ы
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    iteration = 0
    epoch_offset = 0

    # Если есть путь к чекпоинту, загружаем модель
    if checkpoint_path is not None:
        if warm_start:
            # Частичная загрузка весов (игнорируем слои)
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            # Полная загрузка
            model, optimizer, _lr, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _lr
            iteration += 1
            # Определяем, с какой эпохи начинать (при дообучении)
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()

    # Основной цикл по эпохам
    for epoch in range(epoch_offset, hparams.epochs):
        print(f"Epoch: {epoch}")

        # Цикл по батчам
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()

            # Устанавливаем learning_rate для всех групп параметров
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # Обнуляем градиенты
            model.zero_grad()

            # Парсим батч на входы x и цели y
            x, y = model.parse_batch(batch)

            # Основной прямой проход и вычисление лосса с amp
            with autocast(device_type='cuda', enabled=hparams.fp16_run):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            # Усредняем лосс, если распределённый режим
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            # backward c помощью GradScaler
            scaler.scale(loss).backward()

            # unscale для clip_grad_norm
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh
            )

            # step + update GradScaler
            scaler.step(optimizer)
            scaler.update()

            duration = time.perf_counter() - start

            # Логгируем и печатаем только на rank==0
            if rank == 0:
                print(f"Train loss {iteration} {reduced_loss:.6f} GradNorm {grad_norm:.6f} {duration:.2f}s/it")
                if logger is not None:
                    logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration)

            # Периодически делаем валидацию и сохраняем чекпоинт
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

    # Запуск обучения
    train(args.output_directory, args.log_directory,
          args.checkpoint_path,
          args.warm_start,
          args.n_gpus,
          args.rank,
          args.group_name,
          hparams)
