import sys
import os
import torch
import nltk
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QDoubleSpinBox, QMessageBox, QFileDialog, QLineEdit
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io.wavfile import write

nltk.download('punkt')

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
        self.reference_audio = None

    def initUI(self):
        self.setWindowTitle("XTTS-2 Interface")
        self.setGeometry(300, 100, 600, 700)
        layout = QVBoxLayout()

        # Text input
        layout.addWidget(QLabel("Введите текст для синтеза:"))
        self.text_input = QTextEdit()
        layout.addWidget(self.text_input)

        # Parameter inputs
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

        # Reference audio selection
        layout.addWidget(QLabel("Референсное аудио (опционально):"))
        self.ref_audio_path = QLineEdit()
        browse_ref_button = QPushButton("Выбрать файл")
        browse_ref_button.clicked.connect(self.selectReferenceAudio)
        layout.addWidget(self.ref_audio_path)
        layout.addWidget(browse_ref_button)

        # Buttons
        self.generate_button = QPushButton("Генерировать речь")
        self.generate_button.clicked.connect(self.generateAudio)
        layout.addWidget(self.generate_button)

        self.play_audio_button = QPushButton("Воспроизвести результат")
        self.play_audio_button.clicked.connect(self.playAudio)
        layout.addWidget(self.play_audio_button)

        self.setLayout(layout)
        self.player = QMediaPlayer()

    def loadModel(self):
        self.CONFIG_PATH = "D:/XTTS-v2/config.json"
        self.CHECKPOINT_PATH = "D:/XTTS-v2/"

        # Оптимизация GPU
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        # Load XTTS model
        self.config = XttsConfig()
        self.config.load_json(self.CONFIG_PATH)
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=self.CHECKPOINT_PATH, eval=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).half()  # Переводим модель в FP16

    def selectReferenceAudio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите референсное аудио", "", "WAV Files (*.wav)")
        if file_path:
            self.ref_audio_path.setText(file_path)
            self.reference_audio = file_path

    def generateAudio(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите текст для синтеза.")
            return

        params = {param: widget.value() for param, widget in self.param_widgets.items()}
        processed_text = self.preprocessText(text)

        try:
            with torch.cuda.amp.autocast():  # FP16 для ускорения
                outputs = self.model.synthesize(
                    text=processed_text,
                    config=self.config,
                    speaker_wav=self.reference_audio,
                    language="ru",
                    temperature=params["temperature"],
                    top_k=int(params["top_k"]),
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"],
                    length_penalty=params["length_penalty"],
                    speed=params["speed"],
                    enable_text_splitting=True,
                )

            self.output_counter += 1
            self.current_output_path = os.path.join(os.getcwd(), f"output_{self.output_counter}.wav")

            audio = outputs["wav"]
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            write(self.current_output_path, 24000, np.int16(audio * 32767))
            QMessageBox.information(self, "Готово", f"Результат сохранён как {self.current_output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка генерации", f"Произошла ошибка: {e}")

    def playAudio(self):
        if self.current_output_path and os.path.exists(self.current_output_path):
            url = QUrl.fromLocalFile(self.current_output_path)
            self.player.setMedia(QMediaContent(url))
            self.player.play()
        else:
            QMessageBox.warning(self, "Ошибка", "Нет доступного аудиофайла для воспроизведения.")

    def preprocessText(self, text):
        """
        Обрабатывает текст для создания пауз и интонаций через пунктуацию.
        """
        text = text.replace(",", ", ,")
        text = text.replace(".", ". .")
        text = text.replace("!", "! !")
        text = text.replace("?", "? ?")
        sentences = nltk.sent_tokenize(text)
        return " ".join([sentence.strip().capitalize() for sentence in sentences])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XTTSInterface()
    window.show()
    sys.exit(app.exec_())
