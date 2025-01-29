import logging
from telegram.ext import Application
from config import TOKEN
from handlers import register_handlers

# Настройка логгера
def setup_logging():
    # Общий логгер
    general_logger = logging.getLogger("general")
    general_logger.setLevel(logging.INFO)

    # Обработчик для записи в файл general.log
    general_file_handler = logging.FileHandler("general.log", encoding="utf-8")
    general_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    general_logger.addHandler(general_file_handler)

    # Обработчик для вывода в консоль
    general_console_handler = logging.StreamHandler()
    general_console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    general_logger.addHandler(general_console_handler)

    # Логгер для ошибок
    error_logger = logging.getLogger("error")
    error_logger.setLevel(logging.ERROR)

    # Обработчик для записи в файл error.log
    error_file_handler = logging.FileHandler("error.log", encoding="utf-8")
    error_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    error_logger.addHandler(error_file_handler)

    # Обработчик для вывода ошибок в консоль
    error_console_handler = logging.StreamHandler()
    error_console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    error_logger.addHandler(error_console_handler)

setup_logging()

# Получение логгеров
general_logger = logging.getLogger("general")
error_logger = logging.getLogger("error")

def main():
    general_logger.info("Запуск бота...")
    application = Application.builder().token(TOKEN).build()
    register_handlers(application)
    general_logger.info("Бот успешно запущен")
    application.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_logger.error(f"Критическая ошибка: {e}", exc_info=True)
