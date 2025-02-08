#queues.py
import logging
from collections import defaultdict, deque
from telegram import Update, CallbackQuery

logger = logging.getLogger(__name__)

user_queues = defaultdict(deque)

async def handle_user_task(update_or_query, task_type: str):
    if isinstance(update_or_query, CallbackQuery):
        user_id = update_or_query.from_user.id
    elif isinstance(update_or_query, Update):
        user_id = update_or_query.effective_user.id
    else:
        logger.error("Неизвестный тип данных")
        return

    logger.info(f"Обработка задачи пользователя {user_id}, тип задачи: {task_type}")
    # Пример обработки задачи
    if task_type == "button":
        await update_or_query.edit_message_text(f"Вы нажали кнопку {update_or_query.data}") #убрать текст при подключении нейронки
    elif task_type == "message":
        if update_or_query.message.voice:
            await update_or_query.message.reply_text("Голосовое сообщение обработано.")
        elif update_or_query.message.text:
            await update_or_query.message.reply_text(f"Ваш текст обработан: {update_or_query.message.text}")
