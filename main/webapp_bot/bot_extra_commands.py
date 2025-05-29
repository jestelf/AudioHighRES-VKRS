# bot_extra_commands.py
# Дополнительные команды для Telegram-бота проекта Audio HighRes.

import json
import shutil
from pathlib import Path
from datetime import date
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes

# Пути к базам данных
TARIFFS_DB = "tariffs_db.json"
USERS_EMB = Path("users_emb")
SETTINGS_DB = "user_settings.json"

def load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_tariff(uid: str) -> str:
    tariffs = load_json(TARIFFS_DB)
    return tariffs.get(uid, "free")

def daily_gen_count(uid: str) -> int:
    meta = USERS_EMB / uid / "gen_meta.json"
    if not meta.exists():
        return 0
    d = load_json(meta)
    if d.get("date") != date.today().isoformat():
        return 0
    return d.get("count", 0)

async def cmd_help(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    help_message = (
        "🤖 *Список доступных команд:*\n\n"
        "/start — запуск бота и управление слотами.\n"
        "/tariff — текущий тариф и возможность изменения.\n"
        "/help — вывести это сообщение помощи.\n"
        "/about — информация о проекте.\n"
        "/stats — ваша статистика использования сервиса.\n"
        "/history — ваша история последних запросов.\n"
        "/feedback — отправить отзыв разработчикам.\n\n"
        "Отправьте голос для создания слепка.\n"
        "Отправьте текст, чтобы бот синтезировал речь."
    )
    await upd.message.reply_text(help_message, parse_mode='Markdown')

async def cmd_about(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    about_message = (
        "📌 *Audio HighRes Bot*\n\n"
        "Авторский дипломный проект, предоставляющий функции:\n"
        "— Клонирование голоса и синтез речи.\n"
        "— Антифрод-анализ сообщений.\n"
        "— Управление голосовыми слотами и тарифами.\n\n"
        "Разработчики: Ширшов А.А., Акаев Э.К."
    )
    await upd.message.reply_text(about_message, parse_mode='Markdown')

async def cmd_stats(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = str(upd.effective_user.id)
    tariff = get_tariff(uid)
    daily_gen = daily_gen_count(uid)
    
    stats_message = (
        f"📊 *Статистика использования:*\n\n"
        f"Ваш текущий тариф: *{tariff.title()}*\n"
        f"Голосовых генераций за сегодня: *{daily_gen}*"
    )
    await upd.message.reply_text(stats_message, parse_mode='Markdown')

async def cmd_feedback(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    feedback_text = ' '.join(ctx.args)
    if not feedback_text:
        await upd.message.reply_text("Используйте команду так: /feedback ваш отзыв.")
        return
    uid = str(upd.effective_user.id)
    Path("feedbacks").mkdir(exist_ok=True)
    with open(f"feedbacks/{uid}.txt", "a", encoding="utf-8") as f:
        f.write(feedback_text + "\n")
    await upd.message.reply_text("✅ Ваш отзыв отправлен. Спасибо!")

async def cmd_history(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = str(upd.effective_user.id)
    log_file = USERS_EMB / uid / "message.log"
    if not log_file.exists():
        await upd.message.reply_text("📂 У вас пока нет истории.")
        return
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[-5:]
    history_text = "📝 *Последние запросы:*\n\n" + ''.join(lines)
    await upd.message.reply_text(history_text, parse_mode='Markdown')

