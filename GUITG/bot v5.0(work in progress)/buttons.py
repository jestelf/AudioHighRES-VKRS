from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Меню с кнопками
menus = {
    "main_menu": [
        {"text": "Команда 1", "callback_data": "button_1"},
        {"text": "Команда 2", "callback_data": "button_2"},
        {"text": "Команда 3", "callback_data": "button_3"},
        {"text": "Информация", "callback_data": "info"},
    ],
    "help_menu": [
        {"text": "Помощь 1", "callback_data": "help_1"},
        {"text": "Помощь 2", "callback_data": "help_2"},
        {"text": "Помощь 3", "callback_data": "help_3"},
    ],
    "info_menu": [
        {"text": "О боте", "callback_data": "about"},
        {"text": "Правила", "callback_data": "rules"},
        {"text": "Контакты", "callback_data": "contacts"},
    ],
}

# Функция для создания инлайн кнопок
def create_inline_buttons(menu_name: str):
    if menu_name not in menus:
        raise ValueError(f"Меню с названием {menu_name} не существует")

    buttons = menus[menu_name]
    keyboard = [[InlineKeyboardButton(button["text"], callback_data=button["callback_data"])] for button in buttons]
    return InlineKeyboardMarkup(keyboard)
