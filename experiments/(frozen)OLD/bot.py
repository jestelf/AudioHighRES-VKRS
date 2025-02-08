# bot.py

import os
import asyncio
import uuid
from telegram.ext import (
    ApplicationBuilder, MessageHandler, filters, ContextTypes,
    CallbackQueryHandler, CommandHandler
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
import logging

from processing import (
    process_audio_initial, process_audio_improved,
    process_reference_audio, synthesize_speech,
    process_text_transcription
)
from config import TELEGRAM_TOKEN, WORKING_DIR

from asyncio import Queue

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Определяем этапы разговора
EDIT_TEXT, SET_PARAMETER = range(2)

# Инициализируем очередь для синтеза
synthesis_queue = Queue()

async def voice_or_audio_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('awaiting_reference_audio'):
        # Обрабатываем сообщение как референсное аудио
        await receive_reference_audio(update, context)
    else:
        # Обрабатываем сообщение как голосовое для распознавания речи
        await voice_message_handler(update, context)

async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Получаем голосовое сообщение или аудио
        voice = update.message.voice or update.message.audio

        if not voice:
            await update.message.reply_text("Пожалуйста, отправьте голосовое сообщение или аудиофайл.")
            return

        # Скачиваем файл голосового сообщения
        file = await context.bot.get_file(voice.file_id)
        file_extension = file.file_path.split('.')[-1]
        unique_id = uuid.uuid4()
        file_path = os.path.join(WORKING_DIR, f'voice_{update.effective_user.id}_{unique_id}.{file_extension}')
        await file.download_to_drive(custom_path=file_path)
        logger.info(f"Скачан файл: {file_path}")

        # Обрабатываем аудио файл с маленькой моделью
        initial_output = process_audio_initial(file_path)

        if initial_output is None:
            await update.message.reply_text("Произошла ошибка при обработке аудио.")
            return

        # Отправляем первичную транскрипцию пользователю
        sent_message = await update.message.reply_text(initial_output)
        logger.info("Первичная транскрипция отправлена пользователю.")

        # Сохраняем идентификаторы сообщения и чата в контексте пользователя
        context.user_data['chat_id'] = update.effective_chat.id
        context.user_data['message_id'] = sent_message.message_id

        # Обрабатываем аудио файл с большой моделью
        improved_output = process_audio_improved(file_path)

        if improved_output is None:
            await context.bot.edit_message_text(
                chat_id=context.user_data['chat_id'],
                message_id=context.user_data['message_id'],
                text="Произошла ошибка при улучшенной обработке аудио."
            )
            return

        # Кнопки
        keyboard = [
            [
                InlineKeyboardButton("Редактировать текст", callback_data='edit_text'),
                InlineKeyboardButton("Синтезировать речь", callback_data='synthesize_speech'),
            ],
            [
                InlineKeyboardButton("Настройки синтеза", callback_data='synthesis_settings'),
                InlineKeyboardButton("Загрузить референсное аудио", callback_data='upload_reference'),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Обновляем сообщение с улучшенной транскрипцией и добавляем кнопки
        await context.bot.edit_message_text(
            chat_id=context.user_data['chat_id'],
            message_id=context.user_data['message_id'],
            text=improved_output,
            reply_markup=reply_markup
        )
        logger.info("Сообщение обновлено с улучшенной транскрипцией.")

        # Сохраняем улучшенный текст в контексте пользователя
        context.user_data['transcription'] = improved_output

    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'edit_text':
        # Запрашиваем новый текст у пользователя
        await query.edit_message_reply_markup(reply_markup=None)  # Убираем кнопки
        await query.message.reply_text("Пожалуйста, отправьте новый текст для замены.")
        context.user_data['awaiting_edit_text'] = True
    elif query.data == 'synthesize_speech':
        # Добавляем запрос в очередь
        user_id = query.from_user.id
        chat_id = query.message.chat_id
        transcription = context.user_data.get('transcription')
        if transcription is None:
            await query.message.reply_text("Нет текста для синтеза речи.")
            return

        # Проверяем, есть ли у пользователя референсное аудио
        reference_audio = context.user_data.get('reference_audio')

        # Генерируем уникальный ID для запроса
        request_id = uuid.uuid4()

        # Добавляем запрос в очередь
        await synthesis_queue.put({
            'request_id': str(request_id),
            'user_id': user_id,
            'chat_id': chat_id,
            'transcription': transcription,
            'reference_audio': reference_audio,
            'tts_settings': context.user_data.get('tts_settings', {})
        })

        position_in_queue = synthesis_queue.qsize()
        await query.message.reply_text(f"Ваш запрос добавлен в очередь на синтез речи. Позиция в очереди: {position_in_queue}")
        logger.info(f"Пользователь {user_id} добавлен в очередь на синтез речи с ID {request_id}.")

        # Запускаем обработку очереди, если она не запущена
        if not context.bot_data.get('synthesis_queue_running'):
            context.bot_data['synthesis_queue_running'] = True
            asyncio.create_task(process_synthesis_queue(context))
    elif query.data == 'upload_reference':
        await query.message.reply_text("Пожалуйста, отправьте аудиофайл для использования в качестве референсного аудио.")
        context.user_data['awaiting_reference_audio'] = True
    elif query.data == 'synthesis_settings':
        # Предлагаем параметры для изменения
        keyboard = [
            [
                InlineKeyboardButton("Скорость (1.0)", callback_data='set_speed'),
            ],
            [
                InlineKeyboardButton("Коэффициент повторений (2.0)", callback_data='set_repetition_penalty'),
            ],
            [
                InlineKeyboardButton("Коэффициент длины (1.0)", callback_data='set_length_penalty'),
            ],
            [
                InlineKeyboardButton("Температура (0.7)", callback_data='set_temperature'),
            ],
            [
                InlineKeyboardButton("Назад", callback_data='back_to_main')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Настройки синтеза речи. Нажмите на параметр для изменения:", reply_markup=reply_markup)
    elif query.data == 'set_speed':
        # Предлагаем варианты скорости
        keyboard = [
            [
                InlineKeyboardButton("Медленная (0.8)", callback_data='speed_0.8'),
                InlineKeyboardButton("Нормальная (1.0)", callback_data='speed_1.0'),
            ],
            [
                InlineKeyboardButton("Быстрая (1.2)", callback_data='speed_1.2'),
                InlineKeyboardButton("Назад", callback_data='synthesis_settings')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text(
            "Скорость синтеза речи:\n"
            "Медленная (0.8) - медленнее стандартной скорости.\n"
            "Нормальная (1.0) - стандартная скорость.\n"
            "Быстрая (1.2) - быстрее стандартной скорости.",
            reply_markup=reply_markup
        )
    elif query.data.startswith('speed_'):
        try:
            speed = float(query.data.split('_')[1])
            context.user_data.setdefault('tts_settings', {})['speed'] = speed
            await query.message.reply_text(
                f"Скорость установлена на {speed}.\n\n"
                f"**Скорость:** Определяет, насколько быстро будет воспроизводиться речь. (1.0 — нормальная скорость)",
                parse_mode='Markdown'
            )
            logger.info(f"Пользователь {query.from_user.id} установил скорость на {speed}.")
        except ValueError:
            await query.message.reply_text("Ошибка при установке скорости.")
    elif query.data == 'set_repetition_penalty':
        # Предлагаем варианты коэффициента повторений
        keyboard = [
            [
                InlineKeyboardButton("Низкий (1.5)", callback_data='repetition_penalty_1.5'),
                InlineKeyboardButton("Средний (2.0)", callback_data='repetition_penalty_2.0'),
            ],
            [
                InlineKeyboardButton("Высокий (2.5)", callback_data='repetition_penalty_2.5'),
                InlineKeyboardButton("Назад", callback_data='synthesis_settings')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text(
            "Коэффициент повторений:\n"
            "Низкий (1.5) - меньше повторений.\n"
            "Средний (2.0) - стандартный коэффициент.\n"
            "Высокий (2.5) - больше повторений.",
            reply_markup=reply_markup
        )
    elif query.data.startswith('repetition_penalty_'):
        try:
            repetition_penalty = float(query.data.split('_')[2])
            context.user_data.setdefault('tts_settings', {})['repetition_penalty'] = repetition_penalty
            await query.message.reply_text(
                f"Коэффициент повторений установлен на {repetition_penalty}.\n\n"
                f"**Повторения:** Контролирует, насколько часто модель может повторять слова или фразы. (2.0 — стандартная частота)",
                parse_mode='Markdown'
            )
            logger.info(f"Пользователь {query.from_user.id} установил коэффициент повторений на {repetition_penalty}.")
        except (IndexError, ValueError):
            await query.message.reply_text("Ошибка при установке коэффициента повторений.")
    elif query.data == 'set_length_penalty':
        # Предлагаем варианты коэффициента длины
        keyboard = [
            [
                InlineKeyboardButton("Низкий (0.8)", callback_data='length_penalty_0.8'),
                InlineKeyboardButton("Средний (1.0)", callback_data='length_penalty_1.0'),
            ],
            [
                InlineKeyboardButton("Высокий (1.2)", callback_data='length_penalty_1.2'),
                InlineKeyboardButton("Назад", callback_data='synthesis_settings')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text(
            "Коэффициент длины:\n"
            "Низкий (0.8) - короткие фразы.\n"
            "Средний (1.0) - стандартный коэффициент.\n"
            "Высокий (1.2) - длинные фразы.",
            reply_markup=reply_markup
        )
    elif query.data.startswith('length_penalty_'):
        try:
            length_penalty = float(query.data.split('_')[2])
            context.user_data.setdefault('tts_settings', {})['length_penalty'] = length_penalty
            await query.message.reply_text(
                f"Коэффициент длины установлен на {length_penalty}.\n\n"
                f"**Длина:** Влияет на длину генерируемых предложений. (1.0 — стандартная длина)",
                parse_mode='Markdown'
            )
            logger.info(f"Пользователь {query.from_user.id} установил коэффициент длины на {length_penalty}.")
        except (IndexError, ValueError):
            await query.message.reply_text("Ошибка при установке коэффициента длины.")
    elif query.data == 'set_temperature':
        # Предлагаем варианты температуры
        keyboard = [
            [
                InlineKeyboardButton("Низкая (0.5)", callback_data='temperature_0.5'),
                InlineKeyboardButton("Средняя (0.7)", callback_data='temperature_0.7'),
            ],
            [
                InlineKeyboardButton("Высокая (0.9)", callback_data='temperature_0.9'),
                InlineKeyboardButton("Назад", callback_data='synthesis_settings')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text(
            "Температура генерации:\n"
            "Низкая (0.5) - более детерминированный и стабильный звук.\n"
            "Средняя (0.7) - сбалансированный звук.\n"
            "Высокая (0.9) - более разнообразный и креативный звук.",
            reply_markup=reply_markup
        )
    elif query.data.startswith('temperature_'):
        try:
            temperature = float(query.data.split('_')[1])
            context.user_data.setdefault('tts_settings', {})['temperature'] = temperature
            await query.message.reply_text(
                f"Температура установлена на {temperature}.\n\n"
                f"**Температура:** Определяет степень случайности в генерации речи. (0.7 — сбалансированная случайность)",
                parse_mode='Markdown'
            )
            logger.info(f"Пользователь {query.from_user.id} установил температуру на {temperature}.")
        except (IndexError, ValueError):
            await query.message.reply_text("Ошибка при установке температуры.")
    elif query.data == 'back_to_main':
        # Возвращаемся в главное меню
        keyboard = [
            [
                InlineKeyboardButton("Редактировать текст", callback_data='edit_text'),
                InlineKeyboardButton("Синтезировать речь", callback_data='synthesize_speech'),
            ],
            [
                InlineKeyboardButton("Настройки синтеза", callback_data='synthesis_settings'),
                InlineKeyboardButton("Загрузить референсное аудио", callback_data='upload_reference'),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Возвращаемся в главное меню.", reply_markup=reply_markup)

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('awaiting_edit_text'):
        await receive_new_text(update, context)
    elif context.user_data.get('awaiting_parameter'):
        await receive_parameter_value(update, context)
    else:
        # Обработка обычного текстового сообщения как новой транскрипции
        try:
            new_text = update.message.text.strip()
            if not new_text:
                await update.message.reply_text("Пожалуйста, отправьте непустое текстовое сообщение.")
                return

            # Предобработка текста: добавление пунктуации и капитализация
            punctuated_text = process_text_transcription(new_text)

            # Отправляем обработанный текст пользователю
            sent_message = await update.message.reply_text(punctuated_text)
            logger.info("Транскрипция из текстового сообщения отправлена пользователю.")

            # Сохраняем идентификаторы сообщения и чата в контексте пользователя
            context.user_data['chat_id'] = update.effective_chat.id
            context.user_data['message_id'] = sent_message.message_id

            # Кнопки
            keyboard = [
                [
                    InlineKeyboardButton("Редактировать текст", callback_data='edit_text'),
                    InlineKeyboardButton("Синтезировать речь", callback_data='synthesize_speech'),
                ],
                [
                    InlineKeyboardButton("Настройки синтеза", callback_data='synthesis_settings'),
                    InlineKeyboardButton("Загрузить референсное аудио", callback_data='upload_reference'),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Отправляем сообщение с кнопками
            await context.bot.edit_message_text(
                chat_id=context.user_data['chat_id'],
                message_id=context.user_data['message_id'],
                text=punctuated_text,
                reply_markup=reply_markup
            )
            logger.info("Сообщение обновлено с транскрипцией из текстового сообщения.")

            # Сохраняем транскрипцию
            context.user_data['transcription'] = punctuated_text

        except Exception as e:
            logger.error(f"Произошла ошибка при обработке текстового сообщения: {str(e)}")
            await update.message.reply_text(f"Произошла ошибка при обработке текста: {str(e)}")

async def receive_new_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_text = update.message.text
    chat_id = context.user_data.get('chat_id')
    message_id = context.user_data.get('message_id')

    if not chat_id or not message_id:
        await update.message.reply_text("Не удалось найти сообщение для редактирования.")
        return

    # Предобработка нового текста: добавление пунктуации и капитализация
    punctuated_text = process_text_transcription(new_text)

    # Обновляем сообщение с новым текстом и кнопками
    keyboard = [
        [
            InlineKeyboardButton("Редактировать текст", callback_data='edit_text'),
            InlineKeyboardButton("Синтезировать речь", callback_data='synthesize_speech'),
        ],
        [
            InlineKeyboardButton("Настройки синтеза", callback_data='synthesis_settings'),
            InlineKeyboardButton("Загрузить референсное аудио", callback_data='upload_reference'),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.edit_message_text(
        chat_id=chat_id,
        message_id=message_id,
        text=punctuated_text,
        reply_markup=reply_markup
    )
    context.user_data['transcription'] = punctuated_text
    context.user_data['awaiting_edit_text'] = False
    logger.info("Сообщение обновлено с новым текстом от пользователя.")

async def receive_parameter_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    param = context.user_data.get('awaiting_parameter')
    value = update.message.text
    try:
        value = float(value)
        if param == 'speed':
            context.user_data.setdefault('tts_settings', {})['speed'] = value
        elif param == 'repetition_penalty':
            context.user_data.setdefault('tts_settings', {})['repetition_penalty'] = value
        elif param == 'length_penalty':
            context.user_data.setdefault('tts_settings', {})['length_penalty'] = value
        elif param == 'temperature':
            context.user_data.setdefault('tts_settings', {})['temperature'] = value
        await update.message.reply_text(f"Параметр '{param}' установлен на {value}.")
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите числовое значение.")
    context.user_data['awaiting_parameter'] = None

async def receive_reference_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверяем, ожидает ли бот референсное аудио
    if not context.user_data.get('awaiting_reference_audio'):
        await update.message.reply_text("Пожалуйста, используйте кнопки для взаимодействия или отправьте голосовое сообщение.")
        return

    # Обработка референсного аудио
    audio_file = update.message.audio or update.message.voice
    if not audio_file:
        await update.message.reply_text("Пожалуйста, отправьте аудиофайл.")
        return

    try:
        # Скачиваем файл
        file = await context.bot.get_file(audio_file.file_id)
        file_extension = file.file_path.split('.')[-1]
        unique_id = uuid.uuid4()
        file_path = os.path.join(WORKING_DIR, f'reference_{update.effective_user.id}_{unique_id}.{file_extension}')
        await file.download_to_drive(custom_path=file_path)
        logger.info(f"Референсное аудио скачано: {file_path}")

        # Обрабатываем референсное аудио
        ref_audio_path = process_reference_audio(file_path)
        if ref_audio_path is not None:
            context.user_data['reference_audio'] = ref_audio_path
            await update.message.reply_text("Референсное аудио успешно загружено и обработано.")
        else:
            await update.message.reply_text("Произошла ошибка при обработке референсного аудио.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке референсного аудио: {str(e)}")
        await update.message.reply_text("Произошла ошибка при загрузке аудио.")
    finally:
        context.user_data['awaiting_reference_audio'] = False

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Действие отменено.")
    context.user_data.clear()

async def process_synthesis_queue(context: ContextTypes.DEFAULT_TYPE):
    while not synthesis_queue.empty():
        request = await synthesis_queue.get()
        user_id = request['user_id']
        chat_id = request['chat_id']
        transcription = request['transcription']
        reference_audio = request['reference_audio']
        tts_settings = request['tts_settings']

        # Информируем пользователя о начале синтеза
        await context.bot.send_message(chat_id=chat_id, text="Синтез речи начался, пожалуйста, подождите...")

        audio_paths = []
        variations = [
            {
                'speed': tts_settings.get('speed', 1.0),
                'repetition_penalty': tts_settings.get('repetition_penalty', 2.0),
                'length_penalty': tts_settings.get('length_penalty', 1.0),
                'temperature': tts_settings.get('temperature', 0.7)
            },
            {
                'speed': min(tts_settings.get('speed', 1.0) + 0.1, 2.0),
                'repetition_penalty': tts_settings.get('repetition_penalty', 2.0),
                'length_penalty': tts_settings.get('length_penalty', 1.0),
                'temperature': min(tts_settings.get('temperature', 0.7) + 0.05, 1.0)
            },
            {
                'speed': max(tts_settings.get('speed', 1.0) - 0.1, 0.5),
                'repetition_penalty': tts_settings.get('repetition_penalty', 2.0),
                'length_penalty': tts_settings.get('length_penalty', 1.0),
                'temperature': max(tts_settings.get('temperature', 0.7) - 0.05, 0.5)
            },
        ]

        for idx, variation in enumerate(variations, start=1):
            audio_path = synthesize_speech(transcription, reference_audio, variation)
            if audio_path:
                audio_paths.append((audio_path, idx))
            else:
                await context.bot.send_message(chat_id=chat_id, text=f"Произошла ошибка при синтезе речи варианта {idx}.")
                logger.error(f"Ошибка синтеза речи для пользователя {user_id}, вариант {idx}.")

        if audio_paths:
            try:
                for path, idx in audio_paths:
                    with open(path, 'rb') as audio_file:
                        await context.bot.send_audio(chat_id=chat_id, audio=InputFile(audio_file), caption=f"Вариант {idx}")
                    logger.info(f"Аудиофайл варианта {idx} для пользователя {user_id} отправлен.")

                    # Удаляем синтезированный аудиофайл
                    os.remove(path)
                    logger.info(f"Синтезированный аудиофайл {path} удален.")
            except Exception as e:
                logger.error(f"Ошибка отправки аудио пользователю {user_id}: {str(e)}")
                await context.bot.send_message(chat_id=chat_id, text="Произошла ошибка при отправке синтезированных аудио.")
        else:
            await context.bot.send_message(chat_id=chat_id, text="Произошла ошибка при синтезе речи.")
            logger.error(f"Ошибка синтеза речи для пользователя {user_id}.")

        synthesis_queue.task_done()

        # Обновляем позиции в очереди для остальных пользователей
        await update_queue_positions(context)

    context.bot_data['synthesis_queue_running'] = False

async def update_queue_positions(context: ContextTypes.DEFAULT_TYPE):
    # Обновление позиций в очереди для пользователей
    queue_list = list(synthesis_queue._queue)
    for idx, request in enumerate(queue_list):
        user_id = request['user_id']
        chat_id = request['chat_id']
        position = idx + 1
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"Ваша позиция в очереди на синтез речи: {position}")
        except Exception as e:
            logger.error(f"Ошибка отправки обновления очереди пользователю {user_id}: {str(e)}")

def main():
    # Создаём приложение бота
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Обработчик кнопок
    application.add_handler(CallbackQueryHandler(button_handler))

    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    # Обработчик голосовых и аудио сообщений
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_or_audio_message_handler))

    # Обработчик для отмены
    application.add_handler(CommandHandler('cancel', cancel))

    # Запускаем бота
    logger.info("Бот запущен и ожидает сообщений...")
    application.run_polling()

if __name__ == '__main__':
    main()
