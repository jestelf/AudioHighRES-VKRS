# bot.py
import asyncio
from telegram import Update, WebAppInfo, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from pyngrok import ngrok, conf
import subprocess
import sys
import os
import time
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение токенов из переменных окружения
TOKEN = os.getenv('6297587605:AAH_ejse0L50eAuk-7Te2fEGS38tW8E5v3c')
NGROK_AUTH_TOKEN = os.getenv('2sGAJ87gijF86mq9FYivbdFWKHw_7yV9Ef9bAtccFkahwAFR6')

# Проверка наличия токенов
if TOKEN:
    print("Ошибка: TELEGRAM_TOKEN не установлен.")
    sys.exit(1)

# Глобальная переменная для хранения WEB_APP_URL
WEB_APP_URL = 'https://b138-85-192-61-46.ngrok-free.app'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global WEB_APP_URL
    if not WEB_APP_URL:
        await update.message.reply_text("Web App еще не готов. Пожалуйста, попробуйте позже.")
        return

    web_app = WebAppInfo(url=f"{WEB_APP_URL}/")
    await update.message.reply_text(
        'Откройте Web App',
        reply_markup=InlineKeyboardMarkup.from_button(
            InlineKeyboardButton(text='Открыть', web_app=web_app)
        )
    )

def start_flask_app():
    """Запуск Flask-приложения в отдельном процессе."""
    flask_script = os.path.join(os.path.dirname(__file__), "app.py")
    return subprocess.Popen([sys.executable, flask_script])

async def main():
    global WEB_APP_URL

    # Настройка токена ngrok (опционально)
    if NGROK_AUTH_TOKEN:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN

    # Запуск Flask-приложения
    flask_process = start_flask_app()
    print("Запущено Flask-приложение.")

    # Дайте Flask время для запуска
    time.sleep(3)

'''
    try:
        # Создание туннеля ngrok на порт 5000
        tunnel = ngrok.connect(5000, "http")
        public_url = tunnel.public_url
        print(f'ngrok tunnel "{public_url}" -> "http://127.0.0.1:5000"')

        WEB_APP_URL = public_url  # Устанавливаем глобальную переменную

        # Настройка и запуск Telegram бота
        application = ApplicationBuilder().token(TOKEN).build()

        application.add_handler(CommandHandler("start", start))

        print("Запущен Telegram-бот. Ожидание команд...")
        await application.run_polling()

    except Exception as e:
        print(f"Произошла ошибка: {e}")

    finally:
        # Завершение туннеля и Flask-приложения при остановке бота
        print("Завершение туннеля ngrok и Flask-приложения.")
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        flask_process.terminate()
'''
if __name__ == '__main__':
    asyncio.run(main())
