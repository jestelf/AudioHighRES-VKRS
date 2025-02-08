# configs.py
import torch

class Config:
    DATA_ROOT = r"D:\DatasetDusha"
    TRAIN_TSV = r"D:\DatasetDusha\crowd_train\raw_crowd_train.tsv"
    TEST_TSV = r"D:\DatasetDusha\crowd_test\raw_crowd_test.tsv"
    TRAIN_WAVS = r"D:\DatasetDusha\crowd_train\wavs"
    TEST_WAVS = r"D:\DatasetDusha\crowd_test\wavs"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MEL_CHANNELS = 80
    SPEAKER_EMBED_DIM = 128
    EMOTION_EMBED_DIM = 128
    TEXT_EMBED_DIM = 256
    HIDDEN_DIM = 256
    BATCH_SIZE = 4
    LR = 1e-4
    EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
