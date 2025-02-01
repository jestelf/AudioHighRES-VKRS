# -*- coding: utf-8 -*-

"""
train.py - Скрипт обучения HiFi-GAN (или похожей GAN-модели для вокодинга).
Он:
1. Загружает данные и создает даталоадеры (train и validation).
2. Создаёт модель генератора (Generator) и несколько дискриминаторов (MPD, MSD).
3. Настраивает оптимизаторы и планировщики learning rate (scheduler).
4. Организует цикл обучения (по эпохам и итерациям):
   - Прогон батча через генератор и дискриминаторы,
   - Расчёт лоссов (generator_loss, discriminator_loss),
   - Обновление весов,
   - Периодические валидация, сохранение чекпоинтов и логгирование в TensorBoard.
5. Поддерживает распределённое обучение (DDP, torch.distributed), если num_gpus > 1.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

# Настройка cuDNN: benchmark=True, чтобы возможно ускорить работу свёрточных операций
torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    """
    Основная функция обучения (запускается на каждом процессе/GPU в режиме DDP).

    Параметры:
    ----------
    rank : int
        Номер текущего процесса (0, 1, 2, ...).
    a : Namespace
        Аргументы, переданные скрипту (с командной строки).
    h : AttrDict
        Гиперпараметры (загруженные из JSON config).

    Работает в следующем порядке:
    1. Если требуется, инициализирует процесс в распределённом режиме (init_process_group).
    2. Выставляет random seed, создаёт девайс (cuda:rank).
    3. Создаёт модели: Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator.
    4. Если есть чекпоинты, загружает их для продолжения обучения (scan_checkpoint, load_checkpoint).
    5. При необходимости, оборачивает модели в DistributedDataParallel.
    6. Создаёт оптимизаторы (AdamW) и планировщики (ExponentialLR).
    7. Готовит датасет и DataLoader (MelDataset).
    8. Запускает цикл обучения по эпохам, внутри которого:
       - Итерирует по батчам, считает лоссы, делает backward() и step() оптимизаторов.
       - С некоторой периодичностью (summary_interval, checkpoint_interval, validation_interval):
         логгирует лоссы, делает валидацию, сохраняет модель.
    """
    # Если несколько GPU, инициализируем распределённое обучение
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank
        )

    # Фиксируем seed для воспроизводимости
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    # Создаём экземпляры моделей (генератор + два дискриминатора)
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # На rank=0 печатаем структуру и создаём директорию чекпоинтов
    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    # Ищем существующие чекпоинты (если есть)
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')   # генератор
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_') # дискриминаторы + оптимизаторы
    else:
        cp_g = None
        cp_do = None

    steps = 0
    if cp_g is None or cp_do is None:
        # Если чекпоинтов нет, начнём с нуля
        state_dict_do = None
        last_epoch = -1
    else:
        # Если чекпоинты найдены, загружаем их
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        # Восстанавливаем веса
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    # Оборачиваем модели в DDP, если несколько GPU
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    # Создаем оптимизаторы (AdamW)
    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate,
        betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )

    # Если уже была тренировка (state_dict_do не пустой), восстанавливаем оптимизаторы
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    # Планировщики (scheduler) для управления learning rate
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # Получаем список файлов для train/val (WAV -> MEL)
    training_filelist, validation_filelist = get_dataset_filelist(a)

    # Создаём датасет для обучения (MelDataset)
    # shuffle=False если DDP, т.к. используется DistributedSampler
    trainset = MelDataset(
        training_filelist, h.segment_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,  # В DDP shuffle=False, полагаемся на sampler
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True
    )

    # Если это главный процесс (rank=0), создаём валидационный датасет и TensorBoard writer
    if rank == 0:
        validset = MelDataset(
            validation_filelist, h.segment_size, h.n_fft, h.num_mels,
            h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
            shuffle=False, # validation не мешаем
            fmax_loss=h.fmax_for_loss,
            device=device,
            fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir
        )
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True
        )
        # Инициализация логгера (TensorBoard)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    # Переводим модели в режим train
    generator.train()
    mpd.train()
    msd.train()

    # Основной цикл по эпохам
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        # Для DDP нужно выставить seed самплера (epoch)
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        # Цикл по батчам
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch

            # Загружаем на GPU
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)  # (B, 1, T)

            # Генератор
            y_g_hat = generator(x)  # (B, 1, T)
            # Мел-спектр для сгенерированного аудио
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1),
                h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size,
                h.fmin, h.fmax_for_loss
            )

            # ---- Обучение дискриминаторов ----
            optim_d.zero_grad()

            # Multi-Period Discriminator
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # Multi-Scale Discriminator
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            # backward + step дискриминаторов
            loss_disc_all.backward()
            optim_d.step()

            # ---- Обучение генератора ----
            optim_g.zero_grad()

            # L1 Loss по мел-спектрограмме
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # Снова прогоним дискриминаторы (только forward, не detach), чтобы получить fmap
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)

            # Feature Matching Loss для MPD и MSD
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

            # Generator Loss (LSGAN-стиль)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            # Суммарный лосс для генератора
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            loss_gen_all.backward()
            optim_g.step()

            # ---- Логгирование и чекпоинты ----
            if rank == 0:
                if steps % a.stdout_interval == 0:
                    # Печать в STDOUT
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                        format(steps, loss_gen_all, mel_error, time.time() - start_b)
                    )

                # Сохранение чекпоинтов
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()
                        }
                    )
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }
                    )

                # Логгирование в TensorBoard
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", F.l1_loss(y_mel, y_g_hat_mel).item(), steps)

                # Валидация
                if steps % a.validation_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(
                                y_g_hat.squeeze(1),
                                h.n_fft, h.num_mels, h.sampling_rate,
                                h.hop_size, h.win_size,
                                h.fmin, h.fmax_for_loss
                            )
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            # Пишем несколько примеров в TensorBoard
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat.squeeze(1),
                                    h.n_fft, h.num_mels, h.sampling_rate,
                                    h.hop_size, h.win_size,
                                    h.fmin, h.fmax
                                )
                                sw.add_figure(
                                    'generated/y_hat_spec_{}'.format(j),
                                    plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()),
                                    steps
                                )

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        # Шаг шедулеров в конце эпохи
        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    """
    Точка входа в скрипт. Парсит аргументы, загружает конфиг (JSON),
    и запускает процесс обучения (train) в распределённом или обычном режиме.
    """
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    # Основные аргументы: пути к данным, пути к чекпоинтам и т.д.
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    # Читаем JSON-файл с гиперпараметрами
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Создаём окружение (копируем config.json в checkpoint_path, и пр.)
    build_env(a.config, 'config.json', a.checkpoint_path)

    # Фиксируем seed
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        # Делим batch_size на кол-во GPU, если их несколько
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)

    # Если несколько GPU, запускаем mp.spawn для распределённого обучения
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        # Иначе просто запускаем train(0, ...)
        train(0, a, h)


if __name__ == '__main__':
    main()
