import sys
import os
import tempfile
import torch
import nltk
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QDoubleSpinBox, QMessageBox, QFileDialog, QLineEdit
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

# Если хотим показать спектр
import matplotlib
matplotlib.use("Agg")  # <-- Если хотим отрисовывать в окне PyQt, можно попробовать Qt5Agg
                       # Но 'Agg' тоже работает, хотя и без интерактивного окна
import matplotlib.pyplot as plt

try:
    from resemblyzer import VoiceEncoder
    import librosa
    USE_RESEMBLYZER = True
except ImportError:
    USE_RESEMBLYZER = False

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

nltk.download('punkt')


def extract_mel_spectrogram(wav_data, sr=24000):
    """
    Пример псевдо-функции, которая из массива samples извлекает мел-спектр.
    Здесь мы возвращаем просто рандомные данные [Time, MelBins].
    Замените на реальную логику (librosa.feature.melspectrogram).
    """
    dummy_mel = np.random.rand(100, 80).astype(np.float32)
    return dummy_mel

def mel_to_wav(mel_spectrogram):
    """
    Пример псевдо-функции, восстанавливающей аудиосигнал из мел-спектра.
    В реальном проекте используйте реальный вокодер (HiFi-GAN / WaveGlow и т.п.).
    Здесь возвращаем рандомные данные для примера.
    """
    dummy_audio = np.random.rand(24000 * 3).astype(np.float32) * 2 - 1  # 3 секунды псевдо-аудио
    return dummy_audio

def extract_speaker_embedding(file_path):
    """
    Извлекает speaker embedding из аудиофайла через resemblyzer.
    Возвращает вектор np.float32.
    """
    if not USE_RESEMBLYZER:
        raise RuntimeError("Resemblyzer не установлен! (pip install resemblyzer)")

    wav_data, sr = librosa.load(file_path, sr=16000)
    encoder = VoiceEncoder("cuda" if torch.cuda.is_available() else "cpu")
    embedding = encoder.embed_utterance(wav_data)
    embedding = embedding.astype(np.float32)  # Приводим к float32
    return embedding

def embedding_to_wav(embedding):
    """
    Псевдо-функция для «превращения» embedding обратно в звук.
    На практике нужно обучать специальную модель, которая умеет
    из speaker embedding генерировать аудио, либо брать референс
    и частично менять. Здесь просто возвращаем случайный шум.
    """
    dummy_audio = np.random.rand(24000 * 3).astype(np.float32) * 2 - 1
    return dummy_audio

def show_mel_spectrogram_in_window(mel_spectrogram):
    """
    Вывод мел-спектрограммы в отдельном окошке (matplotlib).
    """
    plt.figure(figsize=(6, 4))
    # Обычно mel лежит [Time, MelBins], для наглядности транспонируем
    plt.imshow(mel_spectrogram.T, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.tight_layout()

    # Чтобы показать интерактивно в отдельном окошке, можно сделать plt.show().
    # Но в среде PyQt это иногда конфликтует.
    # Упростим: сохраним во временный файл и откроем, или используем matplotlib.use("Qt5Agg").
    temp_img = os.path.join(tempfile.gettempdir(), "temp_mel_spectrogram.png")
    plt.savefig(temp_img)
    plt.close()

    # Пробуем открыть файл системным вьювером (Windows Photo Viewer или аналог)
    os.startfile(temp_img)

def save_mel_as_png(mel_spectrogram, save_path):
    """
    Сохраняет мел-спектрограмму как PNG (или другой формат).
    """
    # Используем plt.imsave, где mel надо транспонировать, чтобы
    # время шло по горизонтали, а частоты — по вертикали.
    plt.imsave(save_path, mel_spectrogram.T, cmap='magma', origin='lower')


class XTTSInterface(QWidget):
    DEFAULT_PARAMS = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.85,
        "repetition_penalty": 2.0,
        "length_penalty": 1.0,
        "speed": 1.0,
    }

    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        self.output_counter = 0
        self.current_output_path = None

        self.mel_reference_data = None    # np.ndarray для мел-спектра
        self.embedding_reference_data = None  # np.ndarray для speaker embedding
        self.reference_wav_path = None    # str для WAV файла

    def initUI(self):
        self.setWindowTitle("XTTS-2 (Mel/Embedding/WAV) + Show/Save Mel")
        self.setGeometry(300, 100, 700, 850)
        layout = QVBoxLayout()

        # Текст для синтеза
        layout.addWidget(QLabel("Введите текст для синтеза:"))
        self.text_input = QTextEdit()
        layout.addWidget(self.text_input)

        # Параметры синтеза
        layout.addWidget(QLabel("Параметры синтеза:"))
        self.param_widgets = {}
        for param, default_value in self.DEFAULT_PARAMS.items():
            hlayout = QHBoxLayout()
            label = QLabel(param.replace("_", " ").capitalize() + ":")
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-10000.0, 10000.0)
            spinbox.setValue(default_value)
            hlayout.addWidget(label)
            hlayout.addWidget(spinbox)
            layout.addLayout(hlayout)
            self.param_widgets[param] = spinbox

        # ---------- БЛОК 1: Мел-спектр ----------
        layout.addWidget(QLabel("=== Вариант 1: Мел-спектрограмма (Analyze->mel.txt) ==="))
        self.mel_path_line = QLineEdit()
        self.mel_btn_select = QPushButton("Выбрать .txt (мел)")
        self.mel_btn_select.clicked.connect(self.selectMelReference)

        self.mel_btn_analyze = QPushButton("Анализ WAV -> mel.txt")
        self.mel_btn_analyze.clicked.connect(self.analyzeAudio_Mel)

        # Кнопка - Показать мел
        self.mel_btn_show = QPushButton("Показать мел")
        self.mel_btn_show.clicked.connect(self.showCurrentMelSpectrogram)

        # Кнопка - Сохранить мел в PNG
        self.mel_btn_save_png = QPushButton("Сохранить мел в PNG")
        self.mel_btn_save_png.clicked.connect(self.saveCurrentMelAsPng)

        layout.addWidget(self.mel_path_line)
        layout.addWidget(self.mel_btn_select)
        layout.addWidget(self.mel_btn_analyze)
        layout.addWidget(self.mel_btn_show)
        layout.addWidget(self.mel_btn_save_png)

        self.generate_button_mel = QPushButton("Генерировать (mel->wav)")
        self.generate_button_mel.clicked.connect(self.generateAudio_Mel)
        layout.addWidget(self.generate_button_mel)

        # ---------- БЛОК 2: Embedding (fallback) ----------
        layout.addWidget(QLabel("=== Вариант 2: Speaker Embedding -> WAV fallback ==="))
        self.emb_path_line = QLineEdit()
        self.emb_btn_select = QPushButton("Выбрать .txt (embedding)")
        self.emb_btn_select.clicked.connect(self.selectEmbeddingReference)

        self.emb_btn_analyze = QPushButton("Анализ WAV -> emb.txt")
        self.emb_btn_analyze.clicked.connect(self.analyzeAudio_Embedding)

        layout.addWidget(self.emb_path_line)
        layout.addWidget(self.emb_btn_select)
        layout.addWidget(self.emb_btn_analyze)

        self.generate_button_emb = QPushButton("Генерировать (embedding->wav)")
        self.generate_button_emb.clicked.connect(self.generateAudio_EmbeddingFallback)
        layout.addWidget(self.generate_button_emb)

        # ---------- БЛОК 3: WAV напрямую ----------
        layout.addWidget(QLabel("=== Вариант 3: Референсный WAV напрямую ==="))
        self.wav_path_line = QLineEdit()
        self.wav_btn_select = QPushButton("Выбрать WAV")
        self.wav_btn_select.clicked.connect(self.selectWavReference)

        layout.addWidget(self.wav_path_line)
        layout.addWidget(self.wav_btn_select)

        self.generate_button_wav = QPushButton("Генерировать (WAV)")
        self.generate_button_wav.clicked.connect(self.generateAudio_Wav)
        layout.addWidget(self.generate_button_wav)

        # ---------- Кнопка воспроизведения ----------
        self.play_audio_button = QPushButton("Воспроизвести последний результат")
        self.play_audio_button.clicked.connect(self.playAudio)
        layout.addWidget(self.play_audio_button)

        self.setLayout(layout)
        self.player = QMediaPlayer()

    def loadModel(self):
        self.CONFIG_PATH = "D:/XTTS-v2/config.json"
        self.CHECKPOINT_PATH = "D:/XTTS-v2/"

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.config = XttsConfig()
        self.config.load_json(self.CONFIG_PATH)
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=self.CHECKPOINT_PATH, eval=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    # =========== Методы для мел-спектра ===========
    def analyzeAudio_Mel(self):
        """
        Открываем WAV, извлекаем мел, сохраняем в .txt
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите WAV (Mel)", "", "WAV Files (*.wav)")
        if not file_path:
            return
        sr, wav_data = wavfile.read(file_path)
        if wav_data.dtype == np.int16:
            wav_data = wav_data.astype(np.float32) / 32767.0
        mel_spectrogram = extract_mel_spectrogram(wav_data, sr)

        save_path = QFileDialog.getSaveFileName(self, "Сохранить mel.txt", "", "Text Files (*.txt)")
        if save_path[0]:
            try:
                np.savetxt(save_path[0], mel_spectrogram, fmt="%.6f")
                QMessageBox.information(self, "Успех", f"Mel сохранён: {save_path[0]}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения mel", str(e))

    def selectMelReference(self):
        """
        Загружаем mel.txt из файла
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите mel.txt", "", "Text Files (*.txt)")
        if file_path:
            self.mel_path_line.setText(file_path)
            try:
                data = np.loadtxt(file_path, dtype=np.float32)
                self.mel_reference_data = data
            except Exception as e:
                QMessageBox.critical(self, "Ошибка mel", str(e))
                self.mel_reference_data = None

    def showCurrentMelSpectrogram(self):
        """
        Показать мел в отдельном окошке (график) если у нас есть mel_reference_data
        """
        if self.mel_reference_data is None:
            QMessageBox.warning(self, "Предупреждение", "Мел-спектр не загружен!")
            return
        show_mel_spectrogram_in_window(self.mel_reference_data)

    def saveCurrentMelAsPng(self):
        """
        Сохранить мел-спектр в PNG (для визуализации / компрессии)
        """
        if self.mel_reference_data is None:
            QMessageBox.warning(self, "Предупреждение", "Мел-спектр не загружен!")
            return
        save_path = QFileDialog.getSaveFileName(self, "Сохранить мел как PNG", "", "PNG Image (*.png)")
        if save_path[0]:
            try:
                save_mel_as_png(self.mel_reference_data, save_path[0])
                QMessageBox.information(self, "Успех", f"Мел-соображение сохранено в {save_path[0]}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения PNG", str(e))

    def generateAudio_Mel(self):
        """
        Генерация речи, используя мел -> восстанавливаем wav -> speaker_wav
        """
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст.")
            return
        if self.mel_reference_data is None:
            QMessageBox.warning(self, "Ошибка", "Не выбран мел-референс!")
            return

        params = {param: widget.value() for param, widget in self.param_widgets.items()}
        processed_text = self.preprocessText(text)

        recovered_audio = mel_to_wav(self.mel_reference_data)
        temp_wav_path = os.path.join(tempfile.gettempdir(), "temp_speaker_mel.wav")
        write(temp_wav_path, 24000, np.int16(recovered_audio * 32767))

        try:
            with torch.amp.autocast("cuda"):
                outputs = self.model.synthesize(
                    text=processed_text,
                    config=self.config,
                    speaker_wav=temp_wav_path,
                    language="ru",
                    temperature=params["temperature"],
                    top_k=int(params["top_k"]),
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"],
                    length_penalty=params["length_penalty"],
                    speed=params["speed"],
                    enable_text_splitting=True,
                )
            self.saveAndReport(outputs)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка генерации (mel)", str(e))

    # =========== Методы для EMBEDDING (fallback) ===========
    def analyzeAudio_Embedding(self):
        if not USE_RESEMBLYZER:
            QMessageBox.critical(self, "Ошибка", "Resemblyzer не установлен (pip install resemblyzer)")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите WAV для embedding", "", "WAV Files (*.wav)")
        if not file_path:
            return
        try:
            emb = extract_speaker_embedding(file_path)
        except Exception as ex:
            QMessageBox.critical(self, "Ошибка embedding", str(ex))
            return
        save_path = QFileDialog.getSaveFileName(self, "Сохранить emb.txt", "", "Text Files (*.txt)")
        if save_path[0]:
            try:
                np.savetxt(save_path[0], emb, fmt="%.6f")
                QMessageBox.information(self, "Успех", f"Embedding сохранён: {save_path[0]}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения emb", str(e))

    def selectEmbeddingReference(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите emb.txt", "", "Text Files (*.txt)")
        if file_path:
            self.emb_path_line.setText(file_path)
            try:
                data = np.loadtxt(file_path, dtype=np.float32)
                self.embedding_reference_data = data
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки emb", str(e))
                self.embedding_reference_data = None

    def generateAudio_EmbeddingFallback(self):
        """
        Модель не умеет speaker_embedding,
        поэтому делаем embedding -> wav -> speaker_wav
        """
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст.")
            return
        if self.embedding_reference_data is None:
            QMessageBox.warning(self, "Ошибка", "Не выбран embedding-референс!")
            return

        params = {param: widget.value() for param, widget in self.param_widgets.items()}
        processed_text = self.preprocessText(text)

        recovered_audio = embedding_to_wav(self.embedding_reference_data)
        temp_wav_path = os.path.join(tempfile.gettempdir(), "temp_speaker_emb.wav")
        write(temp_wav_path, 24000, np.int16(recovered_audio * 32767))

        try:
            with torch.amp.autocast("cuda"):
                outputs = self.model.synthesize(
                    text=processed_text,
                    config=self.config,
                    speaker_wav=temp_wav_path,
                    language="ru",
                    temperature=params["temperature"],
                    top_k=int(params["top_k"]),
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"],
                    length_penalty=params["length_penalty"],
                    speed=params["speed"],
                    enable_text_splitting=True,
                )
            self.saveAndReport(outputs)
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Ошибка генерации (embedding->wav)",
                f"Не удалось сгенерировать: {e}"
            )

    # =========== Методы для WAV напрямую ===========
    def selectWavReference(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите WAV (референс)", "", "WAV Files (*.wav)")
        if file_path:
            self.wav_path_line.setText(file_path)
            self.reference_wav_path = file_path

    def generateAudio_Wav(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст.")
            return
        if not self.reference_wav_path:
            QMessageBox.warning(self, "Ошибка", "Не выбран WAV-файл!")
            return

        params = {param: widget.value() for param, widget in self.param_widgets.items()}
        processed_text = self.preprocessText(text)

        try:
            with torch.amp.autocast("cuda"):
                outputs = self.model.synthesize(
                    text=processed_text,
                    config=self.config,
                    speaker_wav=self.reference_wav_path,
                    language="ru",
                    temperature=params["temperature"],
                    top_k=int(params["top_k"]),
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"],
                    length_penalty=params["length_penalty"],
                    speed=params["speed"],
                    enable_text_splitting=True,
                )
            self.saveAndReport(outputs)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка генерации (wav)", str(e))

    # =========== Общие ===========
    def saveAndReport(self, outputs):
        self.output_counter += 1
        self.current_output_path = os.path.join(os.getcwd(), f"output_{self.output_counter}.wav")
        audio = outputs["wav"]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        write(self.current_output_path, 24000, np.int16(audio * 32767))
        QMessageBox.information(self, "Готово", f"Результат сохранён: {self.current_output_path}")

    def playAudio(self):
        if self.current_output_path and os.path.exists(self.current_output_path):
            url = QUrl.fromLocalFile(self.current_output_path)
            self.player.setMedia(QMediaContent(url))
            self.player.play()
        else:
            QMessageBox.warning(self, "Ошибка", "Нет доступного аудиофайла для воспроизведения.")

    def preprocessText(self, text):
        text = text.replace(",", ", ,")
        text = text.replace(".", ". .")
        text = text.replace("!", "! !")
        text = text.replace("?", "? ?")
        sentences = nltk.sent_tokenize(text)
        return " ".join([s.strip().capitalize() for s in sentences])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XTTSInterface()
    window.show()
    sys.exit(app.exec_())
