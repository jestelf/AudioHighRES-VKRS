from flask import Flask, render_template, request, jsonify, send_from_directory
import json, os

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['TEMPLATES_AUTO_RELOAD'] = True

DB = "user_settings.json"

# ────────────────────────────────────────────────
#  helpers
# ────────────────────────────────────────────────
def load_db() -> dict:
    try:
        with open(DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_db(data: dict) -> None:
    with open(DB, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ────────────────────────────────────────────────
#  routes
# ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon"
    )

@app.route("/telegram_auth", methods=["POST"])
def telegram_auth():
    """
    Получаем JSON от Telegram‑Login‑Widget.
    В файл authorized_users.txt добавляем id только
    если его там ещё нет.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "id" not in data:
        return jsonify(status="error", message="Нет ID"), 400

    user_id = str(data["id"])
    file_path = "authorized_users.txt"

    # читаем существующие id в множество
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            users = {line.strip() for line in f}
    except FileNotFoundError:
        users = set()

    # дописываем только новый id
    if user_id not in users:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(user_id + "\n")

    return jsonify(status="success", id=user_id), 200


# -------- POST /save_user_settings ------------
@app.route("/save_user_settings", methods=["POST"])
def save_user_settings():
    payload = request.get_json(force=True, silent=True)
    if not payload or "userId" not in payload or "settings" not in payload:
        return jsonify(status="error", message="Неверные данные"), 400

    user_id  = str(payload["userId"])
    settings = payload["settings"]

    db = load_db()
    db[user_id] = settings
    save_db(db)
    return jsonify(status="success", saved=user_id), 200

# -------- GET /get_user_settings?userId=... ---
@app.route("/get_user_settings")
def get_user_settings():
    user_id = request.args.get("userId", type=str)
    if not user_id:
        return jsonify(status="error", message="userId required"), 400

    db = load_db()
    if user_id not in db:
        return jsonify(status="not_found"), 404

    return jsonify(status="success", settings=db[user_id]), 200

# ────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=5000, debug=True)
