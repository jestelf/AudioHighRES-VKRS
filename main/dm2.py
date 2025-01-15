import sys
import os
import wave
import json
import uuid  # Импортируем модуль uuid
from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import logging
from deepmultilingualpunctuation import PunctuationModel
from razdel import sentenize
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io.wavfile import write
import re

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Уровень логирования Vosk
SetLogLevel(0)

# Путь к рабочей директории
WORKING_DIR = r'D:\prdja'

# Абсолютные пути к моделям
VOSK_MODEL_PATH = os.path.join(WORKING_DIR, 'model', 'vosk-model-ru-0.42')
SMALL_VOSK_MODEL_PATH = os.path.join(WORKING_DIR, 'model', 'vosk-model-small-ru-0.22')

# Пути к моделям синтеза речи
CONFIG_PATH = os.path.join(WORKING_DIR, 'XTTS-v2', 'config.json')
CHECKPOINT_PATH = os.path.join(WORKING_DIR, 'XTTS-v2')
VOCAB_PATH = os.path.join(WORKING_DIR, 'XTTS-v2', 'vocab.json')
SPEAKER_PATH = os.path.join(WORKING_DIR, 'XTTS-v2', 'speakers_xtts.pth')

# Глобальные переменные для моделей
small_model = None
large_model = None
punctuation_model = None
tts_model = None
tts_config = None
device = None

def load_models():
    global small_model, large_model, punctuation_model, tts_model, tts_config, device
    # Загружаем модели Vosk
    try:
        small_model = Model(SMALL_VOSK_MODEL_PATH)
        large_model = Model(VOSK_MODEL_PATH)
        logger.info("Обе модели Vosk загружены.")
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей Vosk: {str(e)}")
        sys.exit(1)

    # Инициализируем модель для восстановления пунктуации
    try:
        punctuation_model = PunctuationModel()
        logger.info("PunctuationModel загружена.")
    except Exception as e:
        logger.error(f"Ошибка загрузки PunctuationModel: {str(e)}")
        punctuation_model = None  # Если модель не загрузилась, устанавливаем None

    # Загружаем модель синтеза речи
    try:
        tts_config = XttsConfig()
        tts_config.load_json(CONFIG_PATH)

        tts_model = Xtts.init_from_config(tts_config)
        tts_model.load_checkpoint(tts_config, checkpoint_dir=CHECKPOINT_PATH, eval=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tts_model = tts_model.to(device)
        logger.info("Модель синтеза речи загружена.")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели синтеза речи: {str(e)}")
        tts_model = None

def convert_ogg_to_wav(ogg_path, wav_path):
    try:
        audio = AudioSegment.from_file(ogg_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(wav_path, format="wav")
        logger.info(f"Конвертировано в WAV: {wav_path}")

        # Проверка свойств WAV-файла
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            logger.info(f"Свойства WAV-файла - Каналы: {channels}, Частота: {framerate} Гц, Глубина бит: {sampwidth * 8} бит")
    except Exception as e:
        logger.error(f"Ошибка конвертации аудио: {str(e)}")
        sys.exit(1)

def transcribe_audio(wav_path, model):
    try:
        wf = wave.open(wav_path, "rb")
    except Exception as e:
        logger.error(f"Не удалось открыть WAV-файл: {str(e)}")
        sys.exit(1)

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        logger.error("Аудио должно быть в формате mono WAV с частотой 16000 Гц.")
        wf.close()
        sys.exit(1)

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    transcript = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_dict = json.loads(result)
            transcript += result_dict.get("text", "") + " "

    # Последний фрагмент
    final_result = rec.FinalResult()
    final_dict = json.loads(final_result)
    transcript += final_dict.get("text", "")
    wf.close()
    logger.info("Распознавание речи завершено.")
    return transcript.strip()

def recase_punctuate(text):
    if punctuation_model is None:
        logger.error("PunctuationModel не загружена.")
        return text  # Возвращаем исходный текст без изменений
    try:
        punctuated_text = punctuation_model.restore_punctuation(text)
        logger.info("Пунктуация и регистр добавлены.")
        return punctuated_text
    except Exception as e:
        logger.error(f"Ошибка при восстановлении пунктуации: {str(e)}")
        return text

def capitalize_sentences(text):
    try:
        sentences = [_.text for _ in sentenize(text)]
        capitalized_sentences = [s.capitalize() for s in sentences]
        return ' '.join(capitalized_sentences)
    except Exception as e:
        logger.error(f"Ошибка при капитализации предложений: {str(e)}")
        return text

def process_audio_initial(ogg_path):
    if not os.path.isfile(ogg_path):
        logger.error(f"Файл {ogg_path} не найден.")
        return None

    # Конвертируем OGG в WAV
    wav_path = os.path.join(WORKING_DIR, f'voice_initial_{uuid.uuid4()}.wav')  # Уникальное имя файла
    convert_ogg_to_wav(ogg_path, wav_path)

    # Распознаём речь с помощью маленькой модели
    initial_transcript = transcribe_audio(wav_path, small_model)
    logger.info(f"Первичный распознанный текст: {initial_transcript}")

    # Добавляем пунктуацию и корректируем регистр с помощью PunctuationModel
    punctuated_initial_text = recase_punctuate(initial_transcript)

    # Капитализируем каждое начало предложения
    punctuated_initial_text = capitalize_sentences(punctuated_initial_text)

    # Удаляем временный файл
    try:
        os.remove(wav_path)
        logger.info(f"Временный файл {wav_path} удален.")
    except Exception as e:
        logger.error(f"Ошибка удаления временного файла {wav_path}: {str(e)}")

    return punctuated_initial_text

def process_audio_improved(ogg_path):
    if not os.path.isfile(ogg_path):
        logger.error(f"Файл {ogg_path} не найден.")
        return None

    # Конвертируем OGG в WAV (можно пропустить, если уже конвертировано)
    wav_path = os.path.join(WORKING_DIR, f'voice_improved_{uuid.uuid4()}.wav')  # Уникальное имя файла
    convert_ogg_to_wav(ogg_path, wav_path)

    # Распознаём речь с помощью большой модели
    improved_transcript = transcribe_audio(wav_path, large_model)
    logger.info(f"Улучшенный распознанный текст: {improved_transcript}")

    # Добавляем пунктуацию и корректируем регистр с помощью PunctuationModel
    punctuated_improved_text = recase_punctuate(improved_transcript)

    # Капитализируем каждое начало предложения
    punctuated_improved_text = capitalize_sentences(punctuated_improved_text)

    # Удаляем временный файл
    try:
        os.remove(wav_path)
        logger.info(f"Временный файл {wav_path} удален.")
    except Exception as e:
        logger.error(f"Ошибка удаления временного файла {wav_path}: {str(e)}")

    return punctuated_improved_text

def process_reference_audio(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        # Обрезаем до 15 секунд, если аудио длиннее
        if len(audio) > 15000:
            audio = audio[:15000]
        # Приводим аудио к нужному формату
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        # Сохраняем обработанное аудио с уникальным именем
        ref_audio_path = os.path.join(WORKING_DIR, f'reference_{uuid.uuid4()}.wav')
        audio.export(ref_audio_path, format='wav')
        logger.info(f"Референсное аудио обработано и сохранено: {ref_audio_path}")
        return ref_audio_path
    except Exception as e:
        logger.error(f"Ошибка обработки референсного аудио: {str(e)}")
        return None

def preprocess_text(text):
    try:
        # Добавляем пробел после знаков препинания, если его нет
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        # Разбиваем текст на предложения
        sentences = re.split(r'(?<=[.!?])\s', text)
        # Капитализируем каждое предложение
        sentences = [s.capitalize() for s in sentences]
        # Объединяем обратно
        processed_text = ' '.join(sentences)
        return processed_text
    except Exception as e:
        logger.error(f"Ошибка при обработке текста для синтеза: {str(e)}")
        return text

def synthesize_speech(text, reference_audio=None, tts_settings=None):
    if tts_model is None:
        logger.error("Модель синтеза речи не загружена.")
        return None

    if tts_settings is None:
        tts_settings = {
            'language': 'ru',
            'speed': 1.0,
            'repetition_penalty': 2.0,
            'length_penalty': 1.0,
            'temperature': 0.7,
            'enable_text_splitting': True
        }

    # Предобработка текста
    processed_text = preprocess_text(text)

    try:
        # Если reference_audio равен None, передаем None или пропускаем параметр speaker_wav
        outputs = tts_model.synthesize(
            text=processed_text,
            config=tts_config,
            speaker_wav=reference_audio if reference_audio else None,
            language=tts_settings.get('language', 'ru'),
            speed=tts_settings.get('speed', 1.0),
            repetition_penalty=tts_settings.get('repetition_penalty', 2.0),
            length_penalty=tts_settings.get('length_penalty', 1.0),
            temperature=tts_settings.get('temperature', 0.7),
            enable_text_splitting=tts_settings.get('enable_text_splitting', True)
        )

        # Извлекаем аудио
        audio = outputs["wav"]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Сохраняем аудио с уникальным именем
        output_path = os.path.join(WORKING_DIR, f'output_{uuid.uuid4()}.wav')
        write(output_path, 24000, audio)
        logger.info(f"Аудиофайл синтезирован и сохранён: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {str(e)}")
        return None

# Загружаем модели при импорте модуля
load_models()
