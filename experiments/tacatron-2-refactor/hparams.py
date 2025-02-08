import re

class HParams:
    """
    Класс для хранения и управления гиперпараметрами модели.
    Позволяет задавать параметры при инициализации и обновлять их из строки.
    """

    def __init__(self, **kwargs):
        """
        Инициализация гиперпараметров. 
        Если при создании экземпляра переданы аргументы, они заменят значения по умолчанию.
        """

        # Основные параметры обучения
        self.epochs = 500  # Количество эпох
        self.iters_per_checkpoint = 1000  # Количество итераций до сохранения чекпоинта
        self.seed = 1234  # Фиксация случайного сидирования для воспроизводимости
        self.dynamic_loss_scaling = True  # Сейчас не используется, так как AMP встроен в PyTorch
        self.fp16_run = False  # Использовать ли 16-битную точность (FP16)
        self.distributed_run = False  # Флаг распределенного обучения
        self.dist_backend = "nccl"  # Бэкенд для распределенного обучения (лучший для GPU)
        self.dist_url = "tcp://localhost:54321"  # Адрес для связи между процессами
        self.cudnn_enabled = True  # Разрешить использование CuDNN
        self.cudnn_benchmark = False  # Оптимизация CuDNN для фиксированных входных размеров
        self.ignore_layers = ["embedding.weight"]  # Слои, которые игнорируются при загрузке чекпоинтов

        # Пути к файлам данных
        self.training_files = r"D:\DatasetDusha\crowd_train\raw_crowd_train.tsv"
        self.validation_files = r"D:\DatasetDusha\crowd_test\raw_crowd_test.tsv"
        self.text_cleaners = ["english_cleaners"]  # Очистка текста перед обработкой

        # Аудио параметры
        self.max_wav_value = 32768.0  # Масштабирование амплитуды аудио
        self.sampling_rate = 16000  # Частота дискретизации (изменена с 22050 на 16000)
        self.filter_length = 1024  # Длина окна Фурье-преобразования
        self.hop_length = 256  # Смещение окна STFT
        self.win_length = 1024  # Длина окна анализа STFT
        self.n_mel_channels = 80  # Количество мел-каналов
        self.mel_fmin = 0.0  # Минимальная частота мел-спектрограммы
        self.mel_fmax = 8000.0  # Максимальная частота мел-спектрограммы

        # Параметры модели
        self.n_symbols = 148  # Количество символов в алфавите модели
        self.symbols_embedding_dim = 512  # Размерность векторного представления символов
        self.encoder_kernel_size = 5  # Размер ядра сверточных слоев энкодера
        self.encoder_n_convolutions = 3  # Количество сверточных слоев энкодера
        self.encoder_embedding_dim = 512  # Размерность скрытых представлений в энкодере
        self.n_frames_per_step = 1  # Количество кадров, генерируемых за один шаг
        self.decoder_rnn_dim = 1024  # Количество нейронов в LSTM-декодере
        self.prenet_dim = 256  # Размерность пренета
        self.max_decoder_steps = 1000  # Максимальное количество шагов декодера
        self.gate_threshold = 0.5  # Порог активации выхода декодера
        self.p_attention_dropout = 0.1  # Дропаут для слоя внимания
        self.p_decoder_dropout = 0.1  # Дропаут в декодере
        self.attention_rnn_dim = 1024  # Количество нейронов в рекуррентном слое внимания
        self.attention_dim = 128  # Размерность слоя внимания
        self.attention_location_n_filters = 32  # Количество фильтров в сверточном attention-слое
        self.attention_location_kernel_size = 31  # Размер ядра свертки в attention-слое
        self.postnet_embedding_dim = 512  # Размерность postnet-слоя
        self.postnet_kernel_size = 5  # Размер ядра свертки в postnet
        self.postnet_n_convolutions = 5  # Количество сверточных слоев в postnet

        # Поддержка мультиречевых и эмоциональных моделей
        self.n_speakers = 10  # Количество различных спикеров
        self.speaker_embedding_dim = 64  # Размерность векторного представления спикеров
        self.n_emotions = 5  # Количество эмоций в датасете
        self.emotion_embedding_dim = 32  # Размерность эмбеддинга эмоций

        # Гиперпараметры оптимизации
        self.use_saved_learning_rate = False  # Использовать сохраненное значение lr или нет
        self.learning_rate = 1e-3  # Начальный learning rate
        self.weight_decay = 1e-6  # Коэффициент L2-регуляризации
        self.grad_clip_thresh = 1.0  # Порог отсечения градиентов
        self.batch_size = 64  # Размер мини-батча
        self.mask_padding = True  # Использовать маскирование паддингов

        # Перезапись параметров, переданных при создании объекта
        for k, v in kwargs.items():
            setattr(self, k, v)

    def parse(self, params_string):
        """
        Разбирает строку параметров в формате "param1=val1,param2=val2" и обновляет их в объекте.
        
        Аргументы:
            params_string (str): строка с параметрами.
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
                val_str = val_str.strip()
                
                # Проверяем, существует ли такой параметр
                if not hasattr(self, key):
                    print(f"Warning: HParams has no attribute {key}. Skip.")
                    continue

                old_value = getattr(self, key)

                # Определяем тип переменной
                def str2bool(s):
                    return s.lower() in ("true", "1", "yes")

                if isinstance(old_value, bool):
                    new_val = str2bool(val_str)
                elif isinstance(old_value, int):
                    new_val = int(val_str)
                elif isinstance(old_value, float):
                    new_val = float(val_str)
                else:
                    new_val = val_str  # Оставляем строку без изменений
                
                setattr(self, key, new_val)


def create_hparams(hparams_string=None):
    """
    Создаёт объект HParams и обновляет его параметры из строки, если она передана.
    
    Аргументы:
        hparams_string (str, optional): строка с параметрами, например, "batch_size=32,learning_rate=0.001".
    
    Возвращает:
        HParams: объект с заданными параметрами.
    """
    h = HParams()
    if hparams_string:
        h.parse(hparams_string)
    return h
