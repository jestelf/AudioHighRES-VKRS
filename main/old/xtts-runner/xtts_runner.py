import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io.wavfile import write
import nltk

# Убедитесь, что загружен модуль токенизации предложений
nltk.download('punkt')

# Пути к вашим файлам
CONFIG_PATH = "D:/XTTS-v2/config.json"  # Укажите полный путь к config.json
CHECKPOINT_PATH = "D:/XTTS-v2/"  # Укажите полный путь к model.pth
VOCAB_PATH = "D:/XTTS-v2/vocab.json"  # Укажите полный путь к vocab.json
SPEAKER_PATH = "D:/XTTS-v2/speakers_xtts.pth"  # Укажите полный путь к speakers_xtts.pth
REFERENCE_AUDIO = "D:/XTTS-v2/voice.wav"  # Укажите путь к референсному аудио или None
OUTPUT_PATH = "D:/XTTS-v2/output.wav"  # Путь для сохранения сгенерированного аудио

# Функция для обработки текста
def preprocess_text(text):
    # Исправление заглавных букв в начале предложений
    sentences = nltk.sent_tokenize(text)
    corrected_sentences = [sentence.capitalize() for sentence in sentences]

    # Добавление пауз на основе пунктуации (паузы будут добавлены в текст явно)
    processed_text = ". ".join(corrected_sentences)
    return processed_text

# Текст для синтеза
raw_text = "Одна девочка ушла из дома в лес. в лесу она заблудилась и стала искать дорогу домой. да не нашла, а пришла в лесу к домику."
processed_text = preprocess_text(raw_text)

# Загружаем конфигурацию модели
config = XttsConfig()
config.load_json(CONFIG_PATH)

# Инициализация модели XTTS
model = Xtts.init_from_config(config)

# Загружаем веса модели
model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_PATH, eval=True)

# Переносим модель на GPU, если доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Генерация речи с дополнительными параметрами
outputs = model.synthesize(
    text=processed_text,
    config=config,
    speaker_wav=REFERENCE_AUDIO,  # Используйте None, если не хотите клонировать голос
    language="ru",  # Укажите язык
    speed=1.0,  # Скорость синтеза (1.0 - стандартная, меньше - медленнее, больше - быстрее)
    repetition_penalty=2.0,  # Штраф за повторения
    length_penalty=1.0,  # Штраф за длину (уменьшает длительность при увеличении значения)
    temperature=0.7,  # Температура для управления вероятностью выборки
    enable_text_splitting=True,  # Разделение текста на предложения
)

# Извлечение аудиосигнала из результата
audio = outputs["wav"]  # Извлекаем аудио из словаря по ключу "wav"

# Убедитесь, что аудио — это NumPy-массив
if isinstance(audio, torch.Tensor):
    audio = audio.cpu().numpy()  # Преобразуем в NumPy, если это тензор

# Сохранение результата
write(OUTPUT_PATH, 24000, audio)
print(f"Аудиофайл сохранён как {OUTPUT_PATH}")

