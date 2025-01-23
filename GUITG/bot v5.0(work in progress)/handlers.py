import logging
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, CallbackContext, filters
from buttons import create_inline_buttons
from queues import process_user_task, handle_user_task

logger = logging.getLogger(__name__)

# Регистрация всех обработчиков
def register_handlers(application):
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("info", info_command))
    application.add_handler(CommandHandler("queue", queue_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT | filters.VOICE, text_or_voice_handler))

# Обработчики команд
async def start_command(update: Update, context: CallbackContext):
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /start")
    greeting = """
    Привет! Добро пожаловать!

    Выберите одну из доступных команд ниже или отправьте текст/голосовое сообщение.
    """
    reply_markup = create_inline_buttons("main_menu")
    await update.message.reply_text(greeting, reply_markup=reply_markup)

async def help_command(update: Update, context: CallbackContext):
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /help")
    instructions = """
    Вот как я могу помочь:

    1. Нажмите кнопки под сообщением для выбора команды.
    2. Отправьте текстовое или голосовое сообщение для обработки.
    """
    reply_markup = create_inline_buttons("help_menu")
    await update.message.reply_text(instructions, reply_markup=reply_markup)

async def info_command(update: Update, context: CallbackContext):
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /info")
    info_text = """
    Информация о боте:
    Этот бот помогает вам взаимодействовать с различными командами и функциями. Выберите одну из опций ниже.
    """
    reply_markup = create_inline_buttons("info_menu")
    await update.message.reply_text(info_text, reply_markup=reply_markup)

async def queue_command(update: Update, context: CallbackContext):
    await handle_user_task(update, "queue")

async def cancel_command(update: Update, context: CallbackContext):
    await handle_user_task(update, "cancel")

async def stats_command(update: Update, context: CallbackContext):
    await handle_user_task(update, "stats")

# Обработчик кнопок
async def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    await handle_user_task(query, "button")

# Обработчик текстовых и голосовых сообщений
async def text_or_voice_handler(update: Update, context: CallbackContext):
    await handle_user_task(update, "text_or_voice")
