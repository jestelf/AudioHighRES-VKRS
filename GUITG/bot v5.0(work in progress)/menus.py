from telegram import InlineKeyboardButton, InlineKeyboardMarkup

menus = {
    "main_menu": [
        {"text": "Команда 1", "callback_data": "button_1"},
        {"text": "Команда 2", "callback_data": "button_2"},
        {"text": "Команда 3", "callback_data": 'button_3'},
    ],
}

def create_inline_buttons(menu_name: str) -> InlineKeyboardMarkup:
    if menu_name not in menus:
        raise ValueError(f"Меню с названием {menu_name} не существует")
    buttons = menus[menu_name]
    keyboard = [[InlineKeyboardButton(button["text"], callback_data=button["callback_data"])] for button in buttons]
    return InlineKeyboardMarkup(keyboard)
