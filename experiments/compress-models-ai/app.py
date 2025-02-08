# app.py
from flask import Flask, request, render_template_string, send_file
import torch
import os
import numpy as np
from model import AudioAutoencoder
from compress_decompress import compress_audio, decompress_audio
import torchaudio
import soundfile as sf
from io import BytesIO
import json

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация модели и загрузка чекпоинта
model = AudioAutoencoder(input_channels=1, base_channels=64, latent_dim=256, output_length=16000)
checkpoint_path = "audio_autoencoder.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# HTML шаблон для веб-интерфейса
HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>Audio Autoencoder Web Interface</title>
</head>
<body>
    <h1>Audio Autoencoder Web Interface</h1>
    <h2>Компрессия аудио</h2>
    <form action="/compress" method="post" enctype="multipart/form-data">
        <label for="file">Выберите аудио (.wav):</label>
        <input type="file" name="file" required><br><br>
        <label for="metadata">Метаданные (JSON):</label>
        <input type="text" name="metadata" placeholder='{"sentence": "Пример"}'><br><br>
        <input type="submit" value="Компрессировать">
    </form>
    <h2>Декомпрессия аудио</h2>
    <form action="/decompress" method="post" enctype="multipart/form-data">
        <label for="file">Выберите сжатый файл (.npz):</label>
        <input type="file" name="file" required><br><br>
        <input type="submit" value="Декомпрессировать">
    </form>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/compress', methods=['POST'])
def compress():
    if 'file' not in request.files:
        return "Файл не найден", 400
    file = request.files['file']
    metadata_str = request.form.get('metadata', '')
    # Сохраняем загруженный файл во временный файл
    temp_audio_path = "temp_input.wav"
    file.save(temp_audio_path)
    try:
        latent = compress_audio(model, temp_audio_path, segment_length=16000, device=device)
    except Exception as e:
        return f"Ошибка при сжатии аудио: {e}", 500
    metadata = {}
    if metadata_str:
        try:
            metadata = json.loads(metadata_str)
        except Exception as e:
            metadata = {"info": metadata_str}
    # Сохраняем сжатые данные в BytesIO (формат npz)
    compressed_bytes = BytesIO()
    np.savez(compressed_bytes, latent=latent, metadata=json.dumps(metadata))
    compressed_bytes.seek(0)
    os.remove(temp_audio_path)
    return send_file(compressed_bytes, as_attachment=True, download_name="compressed_file.npz")

@app.route('/decompress', methods=['POST'])
def decompress():
    if 'file' not in request.files:
        return "Файл не найден", 400
    file = request.files['file']
    temp_npz_path = "temp_compressed.npz"
    file.save(temp_npz_path)
    try:
        data = np.load(temp_npz_path, allow_pickle=True)
        latent = data["latent"]
        metadata = json.loads(data["metadata"].item())
    except Exception as e:
        return f"Ошибка при загрузке сжатого файла: {e}", 500
    try:
        recon = decompress_audio(model, latent, segment_length=16000, device=device)
    except Exception as e:
        return f"Ошибка при восстановлении аудио: {e}", 500
    recon_np = recon.numpy()
    output_buffer = BytesIO()
    sf.write(output_buffer, recon_np.T, 16000, format='WAV')
    output_buffer.seek(0)
    os.remove(temp_npz_path)
    return send_file(output_buffer, as_attachment=True, download_name="reconstructed.wav")

if __name__ == "__main__":
    app.run(debug=True)
