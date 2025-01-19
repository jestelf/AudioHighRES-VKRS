import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, CallbackContext, filters
from handlers_4 import button_handler, text_or_voice_handler  # Импорт обработчиков
from buttons_4 import create_inline_buttons  # Импортируем только функцию создания кнопок

from config import TOKEN

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler_general = logging.FileHandler("general.log", encoding="utf-8")
file_handler_general.setLevel(logging.INFO)

file_handler_error = logging.FileHandler("errors.log", encoding="utf-8")
file_handler_error.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler_general.setFormatter(formatter)
file_handler_error.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler_general)
logger.addHandler(file_handler_error)
logger.addHandler(console_handler)

# Функция для команды /start
async def start_command(update: Update, context: CallbackContext) -> None:
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /start")
    greeting = """
    Привет! Добро пожаловать!

    Выберите одну из доступных команд ниже или отправьте текст/голосовое сообщение.
    """
    reply_markup = create_inline_buttons("main_menu")  # Передаем название меню
    await update.message.reply_text(greeting, reply_markup=reply_markup)

# Функция для команды /help
async def help_command(update: Update, context: CallbackContext) -> None:
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /help")
    instructions = """
    Вот как я могу помочь:

    1. Нажмите кнопки под сообщением для выбора команды.
    2. Отправьте текстовое или голосовое сообщение для обработки.
    """
    reply_markup = create_inline_buttons("help_menu")  # Передаем название меню
    await update.message.reply_text(instructions, reply_markup=reply_markup)

# Основная функция запуска бота
def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_handler))  # Обработка инлайн кнопок
    application.add_handler(MessageHandler(filters.TEXT | filters.VOICE, text_or_voice_handler))  # Текст и голосовые сообщения

    logger.info("Бот успешно запущен")
    application.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {e}")
