# app.py
from flask import Flask, send_from_directory, request, jsonify, redirect, url_for
import os
import json
import hmac
import hashlib
from flask_cors import CORS
from urllib.parse import urlencode

app = Flask(__name__, static_folder='static')
CORS(app)  # Разрешить CORS для всех маршрутов

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'users.json')

# Убедитесь, что папка data существует
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

def load_users():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_user(user_data):
    users = load_users()
    # Проверка, есть ли уже пользователь с таким ID
    if any(user['id'] == user_data['id'] for user in users):
        # Обновляем существующего пользователя
        users = [user if user['id'] != user_data['id'] else user_data for user in users]
    else:
        # Добавляем нового пользователя
        users.append(user_data)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/about')
def about():
    return send_from_directory('static', 'about.html')

@app.route('/tariffs')
def tariffs():
    return send_from_directory('static', 'tariffs.html')

@app.route('/contact')
def contact():
    return send_from_directory('static', 'contact.html')

@app.route('/save_user', methods=['POST'])
def save_user_route():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    save_user(data)
    return jsonify({"status": "success", "message": "User data saved"}), 200

@app.route('/telegram_auth', methods=['GET'])
def telegram_auth():
    # Получение параметров из запроса
    auth_data = request.args.to_dict()
    
    # Верификация данных
    if not verify_auth(auth_data):
        return "Авторизация не удалась", 403
    
    # Сохранение данных пользователя
    user = {
        "id": auth_data.get("id"),
        "first_name": auth_data.get("first_name"),
        "last_name": auth_data.get("last_name", ""),
        "username": auth_data.get("username", ""),
        "photo_url": auth_data.get("photo_url", ""),
        "auth_date": auth_data.get("auth_date")
    }
    save_user(user)
    
    # Перенаправление с приветствием
    query_params = {'welcome': 'true', 'name': user['first_name']}
    redirect_url = f"/?{urlencode(query_params)}"
    return redirect(redirect_url)

def verify_auth(auth_data):
    """
    Верификация данных авторизации от Telegram
    """
    token = os.getenv('TELEGRAM_TOKEN')  # Ваш бот-токен
    if not token:
        print("TELEGRAM_TOKEN не установлен в переменных окружения.")
        return False
    secret_key = hashlib.sha256(token.encode()).digest()

    hash_received = auth_data.pop('hash', None)
    if hash_received is None:
        print("Hash отсутствует в данных авторизации.")
        return False

    # Сортировка данных
    sorted_data = sorted(auth_data.items())
    data_check_string = "\n".join([f"{k}={v}" for k, v in sorted_data])

    # Вычисление HMAC-SHA256
    h = hmac.new(secret_key, msg=data_check_string.encode('utf-8'), digestmod=hashlib.sha256)
    hash_calculated = h.digest()
    hash_calculated_hex = hashlib.sha256(hash_calculated).hexdigest()

    # Сравнение хешей
    is_valid = hash_calculated_hex == hash_received
    if not is_valid:
        print("Верификация данных авторизации не прошла.")
    return is_valid

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
