import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from other.config import TOKEN  # Импортируем токен из файла config.py

# Настроим логирование с сохранением в файл
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),  # Логи будут сохраняться в файл bot.log
        logging.StreamHandler()  # Логи будут также выводиться в консоль
    ]
)
logger = logging.getLogger(__name__)

# Функция, которая будет обрабатывать команду /start
async def start(update: Update, context: CallbackContext) -> None:
    # Логируем факт вызова команды /start
    logger.info("Команда /start вызвана пользователем: %s", update.message.from_user.username)

    # Инструкция к сборке стаканов
    instructions = """
    Привет! Я помогу тебе собрать стаканы из Икеи! Вот инструкция:

    1. Извлеките все стаканы и проверьте их целостность.
    2. Убедитесь, что у вас есть все части. Стаканы обычно идут по 6-12 штук в упаковке.
    3. Для сборки стаканов просто аккуратно распакуйте их и начните использовать по назначению!
    
    Важно:
    - Не пытайтесь собирать стаканы, как мебель.
    - Если стаканы пластиковые, не ставьте их в микроволновку.

    Если у вас возникли проблемы, не стесняйтесь задать вопрос!

    Загрузите голосовое сообщение или текстовое для продолжения работы.
    """

    # Отправляем инструкцию пользователю
    await update.message.reply_text(instructions)

# Функция для обработки ошибок
async def error(update: Update, context: CallbackContext) -> None:
    logger.warning('Ошибка "%s" при обработке обновления "%s"', context.error, update)

def main():
    # Создаем объект Application и передаем ему токен из config.py
    application = Application.builder().token(TOKEN).build()

    # Регистрируем обработчик для команды /start
    application.add_handler(CommandHandler("start", start))

    # Регистрируем обработчик ошибок
    application.add_error_handler(error)

    # Запускаем бота
    application.run_polling()

if __name__ == '__main__':
    main()
