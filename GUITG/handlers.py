import logging
from telegram import Update
from telegram.ext import CallbackContext

# Создаем логгер
logger = logging.getLogger(__name__)

# Настройка логирования для general.log (информационные сообщения)
general_logger = logging.FileHandler("general.log", encoding="utf-8")
general_logger.setLevel(logging.INFO)

# Настройка логирования для errors.log (только ошибки)
error_logger = logging.FileHandler("errors.log", encoding="utf-8")
error_logger.setLevel(logging.ERROR)

# Общая настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Уровень для StreamHandler и general.log
    handlers=[
        general_logger,
        error_logger,
        logging.StreamHandler()  # Выводим в консоль
    ]
)

# Обработчик кнопок
async def button_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    try:
        if query.data == "button_1":
            logger.info(f"Пользователь {query.from_user.id} нажал на кнопку 'Команда 1'")
            await query.edit_message_text(text="Вы выбрали: Команда 1")
        elif query.data == "button_2":
            logger.info(f"Пользователь {query.from_user.id} нажал на кнопку 'Команда 2'")
            await query.edit_message_text(text="Вы выбрали: Команда 2")
        else:
            logger.info(f"Пользователь {query.from_user.id} нажал на неизвестную кнопку")
            await query.edit_message_text(text="Неизвестная команда.")
    except Exception as e:
        logger.error(f"Ошибка в обработчике кнопок: {e}")

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
