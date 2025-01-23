import logging
import asyncio
from collections import defaultdict, deque
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, CallbackContext, filters
from config import TOKEN

def setup_logger():
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
    