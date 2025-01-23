from telegram.ext import Application
from handlers import register_handlers
from logger_setup import setup_logger
from config import TOKEN

# Настройка логгера
logger = setup_logger()

# Основная функция запуска бота
def main():
    application = Application.builder().token(TOKEN).build()

    # Регистрация обработчиков
    register_handlers(application)

    logger.info("Бот успешно запущен")
    application.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {e}")
