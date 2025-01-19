import asyncio  # Для управления задержками
from collections import defaultdict
import logging
from telegram import Update
from telegram.ext import CallbackContext
from buttons_4 import create_inline_buttons  # Импортируем только функцию создания кнопок

# Создаем логгер
logger = logging.getLogger(__name__)

# Словарь для отслеживания состояния пользователей
user_states = defaultdict(lambda: False)

# Обработчик кнопок
async def button_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    user_id = query.from_user.id

    await query.answer()

    if user_states[user_id]:
        logger.info(f"Пользователь {user_id} уже выполняет действие. Нажатие кнопки {query.data} игнорируется.")
        return

    user_states[user_id] = True  # Устанавливаем статус в "занят"

    try:
        # Имитация обработки действия
        response_text = {
            "button_1": "Вы выбрали: Команда 1",
            "button_2": "Вы выбрали: Команда 2",
            "button_3": "Вы выбрали: Команда 3",
            "help_1": "Вы выбрали: Помощь 1",
            "help_2": "Вы выбрали: Помощь 2",
            "help_3": "Вы выбрали: Помощь 3"
        }.get(query.data, "Неизвестная команда")

        logger.info(f"Пользователь {user_id} нажал на {query.data or 'неизвестную'} кнопку")

        # В зависимости от команды создаем соответствующие кнопки
        buttons = "help_menu" if query.data.startswith("help") else "main_menu"
        reply_markup = create_inline_buttons(buttons)

        # Проверяем, нужно ли обновлять сообщение
        current_message_text = query.message.text
        if current_message_text != response_text:
            await query.edit_message_text(text=response_text, reply_markup=reply_markup)
        else:
            logger.info("Попытка изменить сообщение на идентичное. Обновление отменено.")
    except Exception as e:
        logger.error(f"Ошибка в обработчике кнопок: {e}")
    finally:
        await asyncio.sleep(2)  # Задержка для предотвращения спама
        user_states[user_id] = False  # Освобождаем пользователя

# Обработчик текстовых и голосовых сообщений
async def text_or_voice_handler(update: Update, context: CallbackContext) -> None:
    try:
        if update.message.voice:
            logger.info(f"Пользователь {update.effective_user.id} отправил голосовое сообщение")
            await update.message.reply_text("Вы отправили голосовое сообщение. Работаю над ним!")
        elif update.message.text:
            logger.info(f"Пользователь {update.effective_user.id} отправил текст: {update.message.text}")
            await update.message.reply_text(f"Вы отправили текст: {update.message.text}")
        else:
            logger.warning(f"Пользователь {update.effective_user.id} отправил неподдерживаемое сообщение")
            await update.message.reply_text("Не могу обработать это сообщение.")
    except Exception as e:
        logger.error(f"Ошибка в обработчике сообщений: {e}")
