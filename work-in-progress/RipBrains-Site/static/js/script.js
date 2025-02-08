// script.js

document.addEventListener("DOMContentLoaded", function() {
    if (window.Telegram.WebApp) {
        const tg = window.Telegram.WebApp;

        // Получение данных пользователя из Telegram
        const user = tg.initDataUnsafe.user;

        if (user) {
            // Отправка данных пользователя на сервер
            fetch('/save_user', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    id: user.id,
                    first_name: user.first_name,
                    last_name: user.last_name || "",
                    username: user.username || "",
                    language_code: user.language_code || ""
                }),
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                if (data.status === "success") {
                    // Показать приветствие или изменить интерфейс
                    const welcomeElement = document.getElementById("welcome");
                    if (welcomeElement) {
                        welcomeElement.innerText = `Добро пожаловать, ${user.first_name}!`;
                    }
                }
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        }

        // Инициализация Web App
        tg.ready();
    }
});
