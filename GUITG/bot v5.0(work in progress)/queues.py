import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Словарь для отслеживания состояния пользователей
user_states = defaultdict(lambda: False)
# Очередь для управления задачами пользователей
user_queues = defaultdict(deque)

async def process_user_task(user_id: int, context):
    while user_queues[user_id]:
        task = user_queues[user_id].popleft()
        try:
            await task()
        except Exception as e:
            logger.error(f"Ошибка при выполнении задачи пользователя {user_id}: {e}")
    user_states[user_id] = False

async def handle_user_task(update_or_query, task_type: str):
    user_id = update_or_query.effective_user.id

    if user_states[user_id]:
        logger.info(f"Пользователь {user_id} уже выполняет задачу. Тип задачи: {task_type}. Добавлено в очередь.")
        user_queues[user_id].append(lambda: execute_task(update_or_query, task_type))
        return

    user_states[user_id] = True
    await execute_task(update_or_query, task_type)
    await process_user_task(user_id, None)

async def execute_task(update_or_query, task_type: str):
    try:
        if task_type == "queue":
            await update_or_query.message.reply_text("Ваша очередь содержит задачи.")
        elif task_type == "cancel":
            user_id = update_or_query.effective_user.id
            user_queues[user_id].clear()
            await update_or_query.message.reply_text("Ваша очередь успешно очищена.")
        elif task_type == "stats":
            queue_length = len(user_queues[update_or_query.effective_user.id])
            await update_or_query.message.reply_text(f"У вас {queue_length} задач(и) в очереди.")
        elif task_type == "button":
            query = update_or_query
            response_text = {
                "button_1": "Вы выбрали: Команда 1",
                "button_2": "Вы выбрали: Команда 2",
                "button_3": "Вы выбрали: Команда 3",
                "help_1": "Вы выбрали: Помощь 1",
                "help_2": "Вы выбрали: Помощь 2",
                "help_3": "Вы выбрали: Помощь 3",
                "info": "Информация о боте",
                "about": "Этот бот создан для демонстрации возможностей Telegram API.",
                "rules": "Основные правила использования: будьте вежливы и соблюдайте правила общения.",
                "contacts": "Для связи с администратором: admin@example.com"
            }.get(query.data, "Неизвестная команда")

            await query.edit_message_text(response_text)
        elif task_type == "text_or_voice":
            if update_or_query.message.voice:
                await update_or_query.message.reply_text("Вы отправили голосовое сообщение. Работаю над ним!")
            elif update_or_query.message.text:
                await update_or_query.message.reply_text(f"Вы отправили текст: {update_or_query.message.text}")
            else:
                await update_or_query.message.reply_text("Не могу обработать это сообщение.")
    except Exception as e:
        logger.error(f"Ошибка при выполнении задачи {task_type}: {e}")
