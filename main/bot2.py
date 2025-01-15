import logging
import os

from aiogram import Bot, Dispatcher, types
from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)
from aiogram.utils import executor

# from aiogram.contrib.fsm_storage.memory import MemoryStorage

# Импортируем функции из dm2.py
from dm2 import (
    process_audio_initial,
    process_audio_improved,
    process_reference_audio,
    synthesize_speech,
    load_models
)

# ---------------------------------------
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ---------------------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------------------
# ИНИЦИАЛИЗАЦИЯ БОТА И ДИСПЕТЧЕРА
# ---------------------------------------
API_TOKEN = "TOKEN"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)  # Dispatcher(bot, storage=MemoryStorage()) если нужна FSM

# ---------------------------------------
# "БАЗА" ПОЛЬЗОВАТЕЛЕЙ
# ---------------------------------------
# Пример структуры:
# {
#   chat_id: {
#       'action': 'recognize'|'reference'|'settings'|'tts'|None,
#       'reference_audio': "путь/к/референсу.wav" | None,
#       'recognized_text': "последний_итоговый_текст" | None,
#       'tts_settings': {
#           'speed': float,
#           'temperature': float
#       }
#   },
#   ...
# }
user_data_dict = {}

# ---------------------------------------
# ГЛАВНОЕ МЕНЮ (Reply Keyboard)
# ---------------------------------------
def get_main_menu():
    buttons = [
        [KeyboardButton("1. Расшифровать")],
        [KeyboardButton("2. Загрузить референс")],
        [KeyboardButton("3. Настройки синтеза")],
        [KeyboardButton("4. Синтез речи")]
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True)

# ---------------------------------------
# INLINE-КНОПКИ (Настройка синтеза)
# ---------------------------------------
def get_tts_settings_inline(speed, temperature):
    """
    Генерируем Inline-клавиатуру для изменения настроек синтеза.
    """
    buttons = [
        [
            InlineKeyboardButton(f"Скорость: {speed:.2f} ↓", callback_data="speed_down"),
            InlineKeyboardButton(f"↑", callback_data="speed_up")
        ],
        [
            InlineKeyboardButton(f"Температура: {temperature:.2f} ↓", callback_data="temp_down"),
            InlineKeyboardButton(f"↑", callback_data="temp_up")
        ],
        [
            InlineKeyboardButton("Закрыть", callback_data="close_settings")
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

# ---------------------------------------
# INLINE-КНОПКА (Синтез речи)
# ---------------------------------------
def get_synthesize_inline():
    """
    Кнопка «Синтез речи» (callback_data="do_tts").
    """
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton("Синтез речи", callback_data="do_tts")]
        ]
    )
    return keyboard

# ---------------------------------------
# ХЕНДЛЕР /start
# ---------------------------------------
@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    chat_id = message.chat.id

    # Создаём/сбрасываем профиль пользователя
    user_data_dict[chat_id] = {
        'action': 'recognize',  # Сразу переходим в режим распознавания
        'reference_audio': None,
        'recognized_text': None,
        'tts_settings': {'speed': 1.0, 'temperature': 0.7}
    }

    text = (
        "Привет! Я бот для распознавания и синтеза речи.\n\n"
        "Сейчас я в режиме распознавания: можете сразу отправить аудио,\n"
        "и я попробую его преобразовать в текст.\n\n"
        "Или воспользуйтесь меню, если хотите:\n"
        "1) Расшифровать\n"
        "2) Загрузить референс\n"
        "3) Настройки синтеза\n"
        "4) Синтез речи\n"
    )
    await message.answer(text, reply_markup=get_main_menu())

# ---------------------------------------
# ХЕНДЛЕР /help
# ---------------------------------------
@dp.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    help_text = (
        "Список возможностей:\n"
        "1) Можно сразу после /start отправить аудио — бот расшифрует.\n"
        "2) Загрузить референс (тогда потом можно синтезировать голос).\n"
        "3) Настройки синтеза (скорость, температура и т.д.).\n"
        "4) Синтез речи (только при наличии референса).\n\n"
        "Если вы отправляете обычный текст (без команды), он сохранится\n"
        "как «распознанный текст» и будет доступен для синтеза.\n"
    )
    await message.answer(help_text)

# ---------------------------------------
# ОБРАБОТКА ТЕКСТА (НЕ КОМАНДА)
# ---------------------------------------
@dp.message_handler(content_types=['text'])
async def handle_text_messages(message: types.Message):
    chat_id = message.chat.id
    user_text = message.text.strip()

    # Если у пользователя нет профиля, создадим
    if chat_id not in user_data_dict:
        user_data_dict[chat_id] = {
            'action': 'recognize',  
            'reference_audio': None,
            'recognized_text': None,
            'tts_settings': {'speed': 1.0, 'temperature': 0.7}
        }

    lower_text = user_text.lower()

    # Проверим, может быть пользователь нажал пункт меню
    if lower_text.startswith("1. расшифровать"):
        user_data_dict[chat_id]['action'] = 'recognize'
        await message.answer("Отправьте голосовое/аудио для распознавания.")
        return

    if lower_text.startswith("2. загрузить референс"):
        user_data_dict[chat_id]['action'] = 'reference'
        await message.answer("Отправьте референс (до 15 секунд). Формат .ogg, .wav, .mp3 и т.п.")
        return

    if lower_text.startswith("3. настройки синтеза"):
        user_data_dict[chat_id]['action'] = 'settings'
        s = user_data_dict[chat_id]['tts_settings']
        await message.answer(
            "Измените параметры синтеза:",
            reply_markup=get_tts_settings_inline(s['speed'], s['temperature'])
        )
        return

    if lower_text.startswith("4. синтез речи"):
        user_data_dict[chat_id]['action'] = 'tts'
        if not user_data_dict[chat_id]['reference_audio']:
            await message.answer("Сначала загрузите референс (пункт 2).")
            return
        if not user_data_dict[chat_id]['recognized_text']:
            await message.answer("Нет текста для синтеза. Сначала распознайте аудио или пришлите текст.")
            return
        # Запускаем синтез
        await synthesize_current_text(message)
        return

    # Если пользователь просто прислал произвольный текст (не меню):
    # Пропускаем распознавание, сразу сохраняем в recognized_text
    user_data_dict[chat_id]['recognized_text'] = user_text
    # Проверяем, есть ли референс
    ref_path = user_data_dict[chat_id]['reference_audio']
    if ref_path:
        # Показываем Inline-кнопку «Синтез речи»
        await message.answer(
            f"Текст сохранён:\n{user_text}",
            reply_markup=get_synthesize_inline()
        )
    else:
        await message.answer(f"Текст сохранён:\n{user_text}\n\n(Референс не загружен, синтез невозможен)")

# ---------------------------------------
# CALLBACK-ХЕНДЛЕР (Настройки и Синтез)
# ---------------------------------------
@dp.callback_query_handler()
async def handle_inline_callbacks(callback_query: types.CallbackQuery):
    chat_id = callback_query.message.chat.id
    data = callback_query.data

    # Если нет данных профиля
    if chat_id not in user_data_dict:
        await callback_query.answer("Нет данных для пользователя.")
        return

    # Если нажата кнопка «Синтез речи»
    if data == "do_tts":
        await callback_query.answer()  # Закроем "часики"
        await do_tts_inline(callback_query)
        return

    # Иначе — настройки
    settings = user_data_dict[chat_id]['tts_settings']
    speed = settings.get('speed', 1.0)
    temp = settings.get('temperature', 0.7)

    if data == 'speed_up':
        speed = min(speed + 0.1, 3.0)
        settings['speed'] = speed
    elif data == 'speed_down':
        speed = max(speed - 0.1, 0.1)
        settings['speed'] = speed
    elif data == 'temp_up':
        temp = min(temp + 0.1, 2.0)
        settings['temperature'] = temp
    elif data == 'temp_down':
        temp = max(temp - 0.1, 0.0)
        settings['temperature'] = temp
    elif data == 'close_settings':
        await callback_query.message.delete()
        await callback_query.answer("Настройки закрыты.")
        return
    else:
        await callback_query.answer("Неизвестное действие.")
        return

    # Обновляем Inline-клавиатуру c новыми значениями
    await callback_query.message.edit_reply_markup(
        reply_markup=get_tts_settings_inline(settings['speed'], settings['temperature'])
    )
    await callback_query.answer("Настройки обновлены!")

async def do_tts_inline(callback_query: types.CallbackQuery):
    """
    Срабатывает при нажатии Inline-кнопки «Синтез речи».
    """
    chat_id = callback_query.message.chat.id
    recognized_text = user_data_dict[chat_id].get('recognized_text')
    ref_path = user_data_dict[chat_id].get('reference_audio')
    tts_settings = user_data_dict[chat_id].get('tts_settings', {})

    if not ref_path:
        await callback_query.answer("Сначала загрузите референс!", show_alert=True)
        return
    if not recognized_text:
        await callback_query.answer("Нет текста для синтеза!", show_alert=True)
        return

    # Начинаем синтез
    await callback_query.message.edit_text("Синтезирую речь...")
    wav_path = synthesize_speech(
        text=recognized_text,
        reference_audio=ref_path,
        tts_settings=tts_settings
    )
    if not wav_path:
        await callback_query.message.edit_text("Ошибка при синтезе речи.")
        return

    # Отправим аудио
    with open(wav_path, "rb") as f:
        await callback_query.message.answer_audio(f, caption="Результат синтеза")

    os.remove(wav_path)
    await callback_query.message.edit_text("Синтез завершён!")

# ---------------------------------------
# ОБРАБОТКА АУДИО
# ---------------------------------------
@dp.message_handler(content_types=['voice', 'audio', 'document'])
async def handle_voice_or_audio(message: types.Message):
    chat_id = message.chat.id

    # Инициализация, если нет
    if chat_id not in user_data_dict:
        user_data_dict[chat_id] = {
            'action': 'recognize',
            'reference_audio': None,
            'recognized_text': None,
            'tts_settings': {'speed': 1.0, 'temperature': 0.7}
        }

    action = user_data_dict[chat_id].get('action', 'recognize')

    # Скачиваем файл из телеги
    file_id = None
    if message.voice:
        file_id = message.voice.file_id
    elif message.audio:
        file_id = message.audio.file_id
    elif message.document:
        file_id = message.document.file_id

    if not file_id:
        await message.answer("Не удалось получить аудиофайл. Попробуйте снова.")
        return

    # Локальное сохранение
    file_path = f"temp_{chat_id}.ogg"
    file_obj = await bot.get_file(file_id)
    await file_obj.download(destination=file_path)

    # Обработка по действию
    if action == 'reference':
        # Загрузка референса
        ref_path = process_reference_audio(file_path)
        if ref_path is None:
            await message.answer("Ошибка обработки референсного аудио.")
        else:
            user_data_dict[chat_id]['reference_audio'] = ref_path
            await message.answer("Референс успешно загружен!")

        os.remove(file_path)
        return

    # Если action == 'recognize' или что-то другое — делаем распознавание
    # 1) Для ускорения: сразу делаем "initial" → "improved" внутри одной логики
    await message.answer("Обрабатываю аудио...")

    # Сначала process_audio_initial
    text_initial = process_audio_initial(file_path)
    if text_initial is None:
        await message.answer("Ошибка при распознавании.")
        os.remove(file_path)
        return

    # Потом process_audio_improved
    text_improved = process_audio_improved(file_path)
    if text_improved is None:
        # Если вдруг ошибка, хотя бы вернём текст_initial
        user_data_dict[chat_id]['recognized_text'] = text_initial
        await message.answer(f"Текст:\n{text_initial}\n\n(Улучшение не удалось)")
        os.remove(file_path)
        return

    # Удалим временный файл
    os.remove(file_path)

    # Итоговый текст
    recognized_text = text_improved
    user_data_dict[chat_id]['recognized_text'] = recognized_text

    # Проверяем референс
    ref_path = user_data_dict[chat_id].get('reference_audio')
    if ref_path:
        # Показываем Inline-кнопку «Синтез речи»
        await message.answer(
            f"Текст:\n{recognized_text}",
            reply_markup=get_synthesize_inline()
        )
    else:
        await message.answer(f"Текст:\n{recognized_text}\n\n(Референс не загружен)")

# ---------------------------------------
# ОТДЕЛЬНАЯ ФУНКЦИЯ СИНТЕЗА
# ---------------------------------------
async def synthesize_current_text(message: types.Message):
    """
    Синтез командой «4. Синтез речи» из меню.
    """
    chat_id = message.chat.id
    recognized_text = user_data_dict[chat_id].get('recognized_text')
    ref_audio_path = user_data_dict[chat_id].get('reference_audio')
    tts_settings = user_data_dict[chat_id].get('tts_settings', {})

    if not recognized_text:
        await message.answer("Нет текста для синтеза.")
        return
    if not ref_audio_path:
        await message.answer("Нет референса для синтеза.")
        return

    synth_msg = await message.answer("Синтезирую речь...")
    wav_path = synthesize_speech(
        text=recognized_text,
        reference_audio=ref_audio_path,
        tts_settings=tts_settings
    )
    if not wav_path:
        await synth_msg.edit_text("Произошла ошибка при синтезе речи.")
        return

    await synth_msg.edit_text("Отправляю результат...")
    with open(wav_path, "rb") as f:
        await message.answer_audio(f, caption="Результат синтеза")
    os.remove(wav_path)
    await synth_msg.edit_text("Синтез завершён!")

# ---------------------------------------
# СТАРТ БОТА
# ---------------------------------------
if __name__ == "__main__":
    load_models()
    logger.info("Модели загружены. Запускаем бота...")
    executor.start_polling(dp, skip_updates=True)
