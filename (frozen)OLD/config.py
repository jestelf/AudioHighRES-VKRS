# config.py

import os

# Токен вашего Telegram бота
TELEGRAM_TOKEN = '7547606520:AAEzsX_6egz1yGdTrZrjq424jqNHljhRWNg'  # ⚠️ Убедитесь, что токен не доступен публично

# Рабочая директория
WORKING_DIR = r'D:\prdja'

# Абсолютные пути к моделям Vosk
VOSK_MODEL_PATH = os.path.join(WORKING_DIR, 'model', 'vosk-model-ru-0.42')
SMALL_VOSK_MODEL_PATH = os.path.join(WORKING_DIR, 'model', 'vosk-model-small-ru-0.22')

# Пути к моделям синтеза речи XTTS
CONFIG_PATH = os.path.join(WORKING_DIR, 'XTTS-v2', 'config.json')
CHECKPOINT_PATH = os.path.join(WORKING_DIR, 'XTTS-v2')
VOCAB_PATH = os.path.join(WORKING_DIR, 'XTTS-v2', 'vocab.json')
SPEAKER_PATH = os.path.join(WORKING_DIR, 'XTTS-v2', 'speakers_xtts.pth')

# Параметры синтеза речи по умолчанию
DEFAULT_TTS_SETTINGS = {
    'speed': 1.0,
    'repetition_penalty': 2.0,
    'length_penalty': 1.0,
    'temperature': 0.7,
}

# Путь к логам
LOGGING_PATH = os.path.join(WORKING_DIR, 'logs', 'bot.log')

# Конфигурация API-сервера
API_SERVER_URL = 'http://<IP-устройства>:8000/receive_data/'  # Замените <IP-устройства> на IP вашего API-сервера
API_SERVER_TOKEN = 'your_api_server_token'  # Добавьте токен для аутентификации, если требуется