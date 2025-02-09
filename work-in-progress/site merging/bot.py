import logging
import asyncio
from aiogram import Bot, Dispatcher, types, F

TOKEN = "6297587605:AAH_ejse0L50eAuk-7Te2fEGS38tW8E5v3c"
WEB_APP_URL = "https://8273-85-192-61-46.ngrok-free.app"  # Укажите реальный URL или ngrok

bot = Bot(token=TOKEN)
dp = Dispatcher()

logging.basicConfig(level=logging.INFO)

@dp.message(F.text == "/start")
async def start(message: types.Message):
    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[  # Правильное объявление клавиатуры
            [types.InlineKeyboardButton(text="Войти через Telegram", url=WEB_APP_URL)]
        ]
    )
    
    await message.answer("Нажмите кнопку ниже для авторизации:", reply_markup=keyboard)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())