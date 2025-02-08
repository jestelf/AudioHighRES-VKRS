import os
import hashlib
import hmac
import json
import secrets
from flask import Flask, request, render_template, make_response, redirect, session

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Секретный ключ для сессии

BOT_TOKEN = "6297587605:AAH_ejse0L50eAuk-7Te2fEGS38tW8E5v3c"

# Генерируем случайный nonce для CSP
nonce = secrets.token_urlsafe(16)

def check_telegram_auth(data):
    """ Проверка подлинности данных из Telegram Login Widget """
    auth_data = data.copy()
    hash_received = auth_data.pop("hash", None)

    if not hash_received:
        return False

    # Формируем строку проверки
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(auth_data.items()))

    # Генерируем секретный ключ
    secret_key = hashlib.sha256(BOT_TOKEN.encode()).digest()

    # Вычисляем хеш
    calculated_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    return calculated_hash == hash_received

@app.after_request
def apply_csp(response):
    """ Настраиваем Content Security Policy (CSP) """
    csp_header = f"script-src 'self' 'nonce-{nonce}' https://telegram.org;"
    response.headers["Content-Security-Policy"] = csp_header
    return response

@app.route("/")
def index():
    """ Главная страница с Telegram Login Widget """
    if "user" in session:
        return redirect("/dashboard")  # Если уже авторизован, перенаправляем
    return render_template("index.html", nonce=nonce)

@app.route("/auth", methods=["GET"])
def auth():
    """ Обрабатываем входные данные Telegram Login Widget """ 
    data = request.args.to_dict()

    if check_telegram_auth(data):
        session["user"] = data  # Сохраняем данные пользователя в сессии
        return redirect("/dashboard")  # Перенаправляем на защищенную страницу
    else:
        return "Ошибка авторизации", 403

@app.route("/dashboard")
def dashboard():
    """ Защищенная страница после входа """
    if "user" not in session:
        return redirect("/")  # Если не авторизован — отправляем на главную
    return f"<h1>Добро пожаловать, {session['user']['first_name']}!</h1> <a href='/logout'>Выйти</a>"

@app.route("/logout")
def logout():
    """ Выход из системы """
    session.pop("user", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
