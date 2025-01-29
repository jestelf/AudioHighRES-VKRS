import logging
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, CallbackContext, filters
from menus import create_inline_buttons
from queues import handle_user_task

logger = logging.getLogger(__name__)

async def start_command(update: Update, context: CallbackContext) -> None:
    logger.info(f"Пользователь {update.effective_user.id} вызвал команду /start")
    greeting = "Добро пожаловать! Выберите одну из команд:"
    reply_markup = create_inline_buttons("main_menu")
    await update.message.reply_text(greeting, reply_markup=reply_markup)

async def button_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    await handle_user_task(query, "button")

async def text_or_voice_handler(update: Update, context: CallbackContext) -> None:
    await handle_user_task(update, "message")

def register_handlers(application):
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT | filters.VOICE, text_or_voice_handler))
