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

# matplotlib + PIL для сохранения/загрузки PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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
    Демонстрационная функция.
    Возвращает рандомный [Time, MelBins].
    В реальном проекте используйте librosa.feature.melspectrogram или другой метод.
    """
    dummy_mel = np.random.rand(100, 80).astype(np.float32)
    return dummy_mel

def mel_to_wav(mel_spectrogram):
    """
    Псевдо-вокодер, возвращаем случайный шум. 
    Замените на настоящий HiFi-GAN / WaveGlow и т.д.
    """
    dummy_audio = np.random.rand(24000 * 3).astype(np.float32) * 2 - 1
    return dummy_audio

def save_mel_as_png(mel_spectrogram, save_path):
    """
    Сохраняет мел-спектр в PNG. 
    Транспонируем, чтобы [Time] шёл по горизонтали, [MelBins] — по вертикали.
    """
    plt.imsave(save_path, mel_spectrogram.T, cmap='magma', origin='lower')

def load_mel_from_png(png_path):
    """
    Загружает PNG как float-массив и делает обратную транспозицию,
    чтобы получить [Time, MelBins].
    """
    img = Image.open(png_path).convert("RGB")  # или "F" если хотим float32
    # img.size = (width, height)
    # Но imread вернёт нам shape (height, width, 3) 
    # Для мел-спектра обычно 1 канал (grayscale), так что лучше сохранять в cmap='gray' 
    # либо работать с RGB -> тогда придётся как-то восстанавливать (R/G/B)
    # Для упрощения предположим, что 1 канал, но сейчас "magma" — многоцветный. 
    # Тогда придётся декодировать цвет (что сложно).
    # 
    # В реальном проекте проще сохранять grayscale, тогда: 
    #   img = Image.open(png_path).convert("L") 
    # и shape (height, width, 1).
    # 
    # Для демонстрации сделаем вид, что просто берём 1 канал. 
    # (в реальном проекте это может быть неточно).

    # Превращаем в numpy
    arr = np.array(img)
    # arr.shape = (height, width, 3), dtype=uint8
    # Нормируем 0..255 -> 0..1
    arr = arr[..., 0].astype(np.float32) / 255.0  # берём только красный канал

    # arr сейчас (height, width), где height ~ MelBins, width ~ Time
    # Транспонируем обратно
    mel = arr.T  # (width, height) -> (Time, MelBins)
    return mel

def embedding_to_wav(embedding):
    """
    Псевдо-преобразование embedding -> wav.
    """
    dummy_audio = np.random.rand(24000 * 3).astype(np.float32) * 2 - 1
    return dummy_audio

def extract_speaker_embedding(file_path):
    """
    Если модель не поддерживает embedding, это просто демо.
    """
    pass  # Оmitted for brevity or see previous examples


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

        self.mel_reference_data = None
        self.reference_wav_path = None

    def initUI(self):
        self.setWindowTitle("Mel as PNG Demo")
        self.setGeometry(300, 100, 600, 700)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Введите текст для синтеза:"))
        self.text_input = QTextEdit()
        layout.addWidget(self.text_input)

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

        # Кнопка: извлечь мел -> PNG
        self.btn_extract_mel = QPushButton("WAV -> mel.txt + (PNG)")
        self.btn_extract_mel.clicked.connect(self.extractMelToTxtAndPng)
        layout.addWidget(self.btn_extract_mel)

        # Кнопка: загрузить мел из PNG
        self.btn_load_mel_png = QPushButton("Загрузить мел из PNG")
        self.btn_load_mel_png.clicked.connect(self.loadMelFromPng)
        layout.addWidget(self.btn_load_mel_png)

        # Кнопка: сгенерировать
        self.btn_generate = QPushButton("Генерировать (mel->wav)")
        self.btn_generate.clicked.connect(self.generateAudioFromMel)
        layout.addWidget(self.btn_generate)

        # WAV напрямую
        layout.addWidget(QLabel("Референсный WAV:"))
        self.wav_line = QLineEdit()
        self.wav_btn = QPushButton("Выбрать WAV")
        self.wav_btn.clicked.connect(self.selectWavReference)
        layout.addWidget(self.wav_line)
        layout.addWidget(self.wav_btn)

        self.btn_generate_wav = QPushButton("Генерировать (WAV напрямую)")
        self.btn_generate_wav.clicked.connect(self.generateAudio_Wav)
        layout.addWidget(self.btn_generate_wav)

        # Кнопка прослушать
        self.play_button = QPushButton("Воспроизвести аудио")
        self.play_button.clicked.connect(self.playAudio)
        layout.addWidget(self.play_button)

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

    def extractMelToTxtAndPng(self):
        """
        1) Выбираем WAV
        2) Извлекаем мел
        3) Сохраняем в TXT
        4) Сохраняем в PNG
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите WAV для анализа", "", "WAV Files (*.wav)")
        if not file_path:
            return

        sr, wav_data = wavfile.read(file_path)
        if wav_data.dtype == np.int16:
            wav_data = wav_data.astype(np.float32)/32767.0

        mel = extract_mel_spectrogram(wav_data, sr)

        # Сохраняем как TXT
        save_txt = QFileDialog.getSaveFileName(self, "Сохранить mel.txt", "", "Text Files (*.txt)")
        if save_txt[0]:
            np.savetxt(save_txt[0], mel, fmt="%.6f")
            QMessageBox.information(self, "Успех", f"Mel сохранён в TXT: {save_txt[0]}")

        # Сохраняем как PNG
        save_png = QFileDialog.getSaveFileName(self, "Сохранить mel.png", "", "Images (*.png)")
        if save_png[0]:
            save_mel_as_png(mel, save_png[0])
            QMessageBox.information(self, "Успех", f"Mel сохранён в PNG: {save_png[0]}")

    def loadMelFromPng(self):
        """
        Загрузка мел-спектра из PNG (картинки).
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите mel.png", "", "PNG Files (*.png);;All Files (*)")
        if not file_path:
            return

        try:
            mel = load_mel_from_png(file_path)
            QMessageBox.information(self, "Успех", f"Mel загружен из {file_path}")
            self.mel_reference_data = mel
        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения PNG", str(e))

    def generateAudioFromMel(self):
        if self.mel_reference_data is None:
            QMessageBox.warning(self, "Ошибка", "Сперва загрузите мел (TXT или PNG)!")
            return

        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст.")
            return

        params = {param: widget.value() for param, widget in self.param_widgets.items()}

        # Превращаем мел обратно в звук (псевдо)
        recovered_audio = mel_to_wav(self.mel_reference_data)
        temp_wav = os.path.join(tempfile.gettempdir(), "temp_mel_speaker.wav")
        write(temp_wav, 24000, np.int16(recovered_audio*32767))

        try:
            with torch.amp.autocast("cuda"):
                outputs = self.model.synthesize(
                    text=text,
                    config=self.config,
                    speaker_wav=temp_wav,
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
            QMessageBox.critical(self, "Ошибка генерации", str(e))

    def selectWavReference(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите WAV", "", "WAV Files (*.wav)")
        if file_path:
            self.wav_line.setText(file_path)
            self.reference_wav_path = file_path

    def generateAudio_Wav(self):
        if not self.reference_wav_path:
            QMessageBox.warning(self, "Ошибка", "Не выбран WAV-файл.")
            return

        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст.")
            return

        params = {param: widget.value() for param, widget in self.param_widgets.items()}
        try:
            with torch.amp.autocast("cuda"):
                outputs = self.model.synthesize(
                    text=text,
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
            QMessageBox.critical(self, "Ошибка генерации (WAV)", str(e))

    def saveAndReport(self, outputs):
        self.output_counter += 1
        self.current_output_path = os.path.join(os.getcwd(), f"output_{self.output_counter}.wav")
        audio = outputs["wav"]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        write(self.current_output_path, 24000, np.int16(audio*32767))
        QMessageBox.information(self, "Готово", f"Сохранено: {self.current_output_path}")

    def playAudio(self):
        if not self.current_output_path or not os.path.exists(self.current_output_path):
            QMessageBox.warning(self, "Ошибка", "Нет файла для воспроизведения.")
            return
        url = QUrl.fromLocalFile(self.current_output_path)
        self.player.setMedia(QMediaContent(url))
        self.player.play()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = XTTSInterface()
    w.show()
    sys.exit(app.exec_())
