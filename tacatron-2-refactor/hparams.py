# hparams.py

import re

class HParams:
    def __init__(self, **kwargs):
        # Дефолтные значения
        self.epochs = 500
        self.iters_per_checkpoint = 1000
        self.seed = 1234
        self.dynamic_loss_scaling = True  # теперь не используется — встроенный amp
        self.fp16_run = False
        self.distributed_run = False
        self.dist_backend = "nccl"
        self.dist_url = "tcp://localhost:54321"
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.ignore_layers = ["embedding.weight"]

        # Data
        self.training_files = r"D:\DatasetDusha\crowd_train\raw_crowd_train.tsv"
        self.validation_files = r"D:\DatasetDusha\crowd_test\raw_crowd_test.tsv"
        self.text_cleaners = ["english_cleaners"]

        # Audio
        self.max_wav_value = 32768.0
        self.sampling_rate = 16000  # Изменено с 22050 на 16000
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        # Model
        self.n_symbols = 148
        self.symbols_embedding_dim = 512
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512
        self.n_frames_per_step = 1
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1
        self.attention_rnn_dim = 1024
        self.attention_dim = 128
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        # Multi-speaker/emotion
        self.n_speakers = 10
        self.speaker_embedding_dim = 64
        self.n_emotions = 5
        self.emotion_embedding_dim = 32

        # Optimization
        self.use_saved_learning_rate = False
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0
        self.batch_size = 64
        self.mask_padding = True

        # Перезапишем поля, если переданы при создании:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def parse(self, params_string):
        """
        Ожидает строку формата: "param1=val1,param2=val2"
        и перезапишет соответствующие поля.
        """
        if not params_string:
            return
        pairs = re.split(r"[,\s]+", params_string)
        for pair in pairs:
            if not pair.strip():
                continue
            if "=" in pair:
                key, val_str = pair.split("=", 1)
                key = key.strip()
                val_str = pair.split("=", 1)[1].strip()
                # Попробуем угадать тип (int, float, bool)
                if not hasattr(self, key):
                    print(f"Warning: HParams has no attribute {key}. Skip.")
                    continue
                old_value = getattr(self, key)
                # Определим тип:
                def str2bool(s):
                    return s.lower() in ("true", "1", "yes")

                if isinstance(old_value, bool):
                    new_val = str2bool(val_str)
                elif isinstance(old_value, int):
                    new_val = int(val_str)
                elif isinstance(old_value, float):
                    new_val = float(val_str)
                else:
                    # строка
                    new_val = val_str
                setattr(self, key, new_val)


def create_hparams(hparams_string=None):
    """Создание параметров без использования tf.contrib"""
    h = HParams()
    if hparams_string:
        h.parse(hparams_string)
    return h
