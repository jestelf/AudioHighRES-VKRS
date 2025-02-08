# buttons_4.py

# Меню с кнопками
menus = {
    "main_menu": [
        {"text": "Команда 1", "callback_data": "button_1"},
        {"text": "Команда 2", "callback_data": "button_2"},
        {"text": "Команда 3", "callback_data": "button_3"},
        
    ],
    "help_menu": [
        {"text": "Помощь 1", "callback_data": "help_1"},
        {"text": "Помощь 2", "callback_data": "help_2"},
        {"text": "Помощь 3", "callback_data": "help_3"},
    ]
}

# Функция для создания инлайн кнопок
def create_inline_buttons(menu_name: str):
    """
    Функция для создания инлайн кнопок.
    Принимает название меню (например, "main_menu" или "help_menu")
    и возвращает клавиатуру с кнопками.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    if menu_name not in menus:
        raise ValueError(f"Меню с названием {menu_name} не существует")

    buttons = menus[menu_name]
    keyboard = [[InlineKeyboardButton(button["text"], callback_data=button["callback_data"])] for button in buttons]
    return InlineKeyboardMarkup(keyboard)
