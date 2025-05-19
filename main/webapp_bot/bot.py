"""
Mini‑app‑бот + LocalTunnel
--------------------------
1)  BOT_TOKEN  — в переменных окружения или .env
2)  (необ.) LT_SUBDOMAIN="audiohighres"
3)  (необ.) LT_CMD="C:/…/lt.cmd"  или  "lt"  (если в PATH)
4)  python bot.py
"""

import os, re, json, shutil, subprocess
from dotenv import load_dotenv
from telegram import (
    Update, KeyboardButton, ReplyKeyboardMarkup,
    WebAppInfo
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# ────────────────────────────────────────────────────────────
#   Конфигурация
# ────────────────────────────────────────────────────────────
load_dotenv()

BOT_TOKEN: str | None = os.getenv("BOT_TOKEN")
LT_SUBDOMAIN: str = os.getenv("LT_SUBDOMAIN", "audiohighres")
LT_CMD_ENV: str | None = os.getenv("LT_CMD")        # кастомный путь (может быть None)

SETTINGS_DB = "user_settings.json"
AUTH_FILE   = "authorized_users.txt"
WEBAPP_URL: str = ""                                # будет задан при старте

if not os.path.exists(SETTINGS_DB):
    with open(SETTINGS_DB, "w", encoding="utf-8") as f:
        json.dump({}, f)

# ────────────────────────────────────────────────────────────
#   helpers
# ────────────────────────────────────────────────────────────
def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ────────────────────────────────────────────────────────────
#   LocalTunnel
# ────────────────────────────────────────────────────────────
def resolve_lt_cmd() -> str | None:
    """Возвращает путь к lt‑клиенту или None, если не найден"""
    if LT_CMD_ENV:
        return LT_CMD_ENV if os.path.isfile(LT_CMD_ENV) else None
    return shutil.which("lt")                        # в PATH

def start_localtunnel(port: int = 5000) -> str:
    """
    Запускает LocalTunnel и возвращает публичный URL.
    Генерирует исключение, если клиент не найден.
    """
    lt_path = resolve_lt_cmd()
    if not lt_path:
        raise RuntimeError(
            "LocalTunnel CLI не найден.\n"
            "· установите:   npm i -g localtunnel\n"
            "· или укажите переменную LT_CMD с полным путём к lt/lt.cmd"
        )

    cmd = [lt_path, "--port", str(port), "--subdomain", LT_SUBDOMAIN]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        print(line.strip())
        m = re.search(r"https://[a-z0-9\-]+\.loca\.lt", line)
        if m:
            return m.group(0)

    raise RuntimeError("LocalTunnel не выдал публичный URL (сабдомен может быть занят).")

# ────────────────────────────────────────────────────────────
#   Хэндлеры Telegram
# ────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Всегда показываем кнопку Web‑App; ID заносим один раз."""
    user_id = str(update.effective_user.id)

    # --- добавить id в authorized_users.txt (без дублей)
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            known = {line.strip() for line in f}
    except FileNotFoundError:
        known = set()

    if user_id not in known:
        with open(AUTH_FILE, "a", encoding="utf-8") as f:
            f.write(user_id + "\n")

    keyboard = [[KeyboardButton("Открыть форму", web_app=WebAppInfo(url=WEBAPP_URL))]]
    await update.message.reply_text(
        "Откройте форму по кнопке ниже:",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )

async def handle_web_app_data(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    raw = update.message.web_app_data.data
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await update.message.reply_text("❌ Неверный JSON.")
        return

    if payload.get("action") != "save_settings":
        await update.message.reply_text("❌ Неизвестное действие.")
        return

    user_id  = str(update.effective_user.id)
    settings = payload.get("settings", {})

    db = load_json(SETTINGS_DB)
    db[user_id] = settings
    save_json(SETTINGS_DB, db)
    await update.message.reply_text("✅ Настройки сохранены.")

# ────────────────────────────────────────────────────────────
#   main
# ────────────────────────────────────────────────────────────
def main() -> None:
    if not BOT_TOKEN or not re.fullmatch(r"\d+:[\w-]{35}", BOT_TOKEN):
        raise RuntimeError("❌ BOT_TOKEN отсутствует или некорректен.")

    global WEBAPP_URL
    print("⏳ Запускаем LocalTunnel…")
    WEBAPP_URL = start_localtunnel()
    print("✅ LocalTunnel готов:", WEBAPP_URL)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_web_app_data))

    print("🤖 Бот запущен.  Ctrl+C — остановить.")
    app.run_polling()

if __name__ == "__main__":
    main()
