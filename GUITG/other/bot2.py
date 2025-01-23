import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, CallbackContext, filters
from handlers import button_handler, text_or_voice_handler
from other.config import TOKEN

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

# Функция для команды /start
async def start_command(update: Update, context: CallbackContext) -> None:
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /start")
    greeting = """
    Привет! Добро пожаловать!

    Для начала работы отправьте голосовое сообщение или текст.
    Если у вас есть дополнительные вопросы, используйте команду /help.
    """
    keyboard = [["Команда 1", "Команда 2"], ["Команда 3"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(greeting, reply_markup=reply_markup)

# Функция для команды /help
async def help_command(update: Update, context: CallbackContext) -> None:
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /help")
    instructions = """
    Вот как я могу помочь:

    1. Нажмите кнопки в меню для получения быстрых ответов.
    2. Отправьте голосовое сообщение или текст для работы.
    3. Используйте команды:
       - /start — начало работы.
       - /help — это сообщение.

    Если что-то пошло не так, обратитесь за помощью!
    """
    await update.message.reply_text(instructions)

# Основная функция запуска бота
def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_handler))  # Обработка кнопок
    application.add_handler(MessageHandler(filters.TEXT | filters.VOICE, text_or_voice_handler))  # Текст и голосовые сообщения

    logger.info("Бот успешно запущен")
    application.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {e}")
