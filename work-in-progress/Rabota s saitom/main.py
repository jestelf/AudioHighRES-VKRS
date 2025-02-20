TOKEN = ""

import logging
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

# Включаем логирование
logging.basicConfig(level=logging.INFO)

# Создаём экземпляры бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Обработчик команды /start
async def cmd_start(message: Message):
    await message.reply("Привет! Я твой бот.")

# Обработчик команды /help
async def cmd_help(message: Message):
    await message.reply("Доступные команды:\n/start - запустить бота\n/help - справка")

async def main():
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

# Запуск бота
if __name__ == "__main__":
    asyncio.run(main())
