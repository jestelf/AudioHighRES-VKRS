# test_app.py
import os
import csv
import io
import base64
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from configs import Config
from data import build_maps, create_loader
from models import VoiceCloningTTS
from losses import mel_reconstruction_loss, speaker_classification_loss, emotion_classification_loss
from text_preprocessing import normalize_text

def build_vocab_from_tsv(tsv_file):
    texts = []
    with open(tsv_file, "r", encoding="utf-8") as f:
        rr = csv.DictReader(f, delimiter="\t")
        for row in rr:
            st = row["speaker_text"] if row["speaker_text"] else ""
            nt = normalize_text(st)
            texts.append(nt)
    chars = set()
    for t in texts:
        for c in t:
            chars.add(c)
    chs = sorted(list(chars))
    return {c: i+1 for i, c in enumerate(chs)}

def tokenize_batch(txts, c2i):
    if not txts:
        return torch.LongTensor([[]])
    mx = max(len(x) for x in txts)
    out = []
    for t in txts:
        arr = []
        for ch in t:
            arr.append(c2i[ch] if ch in c2i else 0)
        while len(arr) < mx:
            arr.append(0)
        out.append(arr)
    return torch.LongTensor(out)

def load_model_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
    model = VoiceCloningTTS(
        vocab_size=len(ckpt["char_map"])+1,
        n_speakers=len(ckpt["spk_map"])+1,
        n_emotions=len(ckpt["emo_map"])+1
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(Config.DEVICE)
    model.eval()
    return model, ckpt["spk_map"], ckpt["emo_map"], ckpt["char_map"]

def create_confusion_image(y_true, y_pred, title="Confusion", cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def test_checkpoints(ckpt_list):
    if not ckpt_list:
        return ("Не выбраны чекпоинты", "", "")

    smap, emap = build_maps(Config.TEST_TSV)
    char_map = build_vocab_from_tsv(Config.TEST_TSV)
    test_loader = create_loader(
        Config.TEST_TSV,
        Config.TEST_WAVS,
        smap,
        emap,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    total_batches = len(test_loader)

    log_text = "Начинаем тест для выбранных чекпоинтов:\n\n"
    spk_html_all = ""
    emo_html_all = ""

    for ckp_name in ckpt_list:
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckp_name)
        if not os.path.isfile(ckpt_path):
            log_text += f"Файл {ckpt_path} не найден\n"
            continue

        model, spk_map_, emo_map_, c_map = load_model_checkpoint(ckpt_path)
        log_text += f"Чекпоинт [{ckp_name}] загружен.\n"

        # Размерность классов в модели:
        spk_size = len(spk_map_)+1
        emo_size = len(emo_map_)+1

        spk_labels, spk_preds = [], []
        emo_labels, emo_preds = [], []
        mel_losses, spk_losses, emo_losses = [], [], []
        total_samples = 0

        loop = tqdm(test_loader, total=total_batches, ncols=80, desc=f"Eval {ckp_name}")
        for i, batch in enumerate(loop):
            mels = batch["mels"].to(Config.DEVICE, non_blocking=True)
            spk_t = batch["speakers"].to(Config.DEVICE, non_blocking=True)
            emo_t = batch["emotions"].to(Config.DEVICE, non_blocking=True)
            txts = batch["texts"]
            tokens = tokenize_batch(txts, c_map).to(Config.DEVICE, non_blocking=True)

            # Клипуем таргеты, чтобы они были в диапазоне [0, spk_size-1] / [0, emo_size-1]
            # иначе cross_entropy упадёт, если label вышел за пределы.
            spk_t = spk_t.clamp(0, spk_size-1)
            emo_t = emo_t.clamp(0, emo_size-1)

            with torch.no_grad():
                mel_out, wav_out, spk_pred, emo_pred = model(mels, tokens)
                l_mel = mel_reconstruction_loss(mel_out, mels).item()
                l_spk = speaker_classification_loss(spk_pred, spk_t).item()
                l_emo = emotion_classification_loss(emo_pred, emo_t).item()

            mel_losses.append(l_mel)
            spk_losses.append(l_spk)
            emo_losses.append(l_emo)

            spk_hat = torch.argmax(spk_pred, dim=1).cpu().numpy()
            emo_hat = torch.argmax(emo_pred, dim=1).cpu().numpy()
            spk_gt = spk_t.cpu().numpy()
            emo_gt = emo_t.cpu().numpy()

            spk_labels.extend(spk_gt)
            spk_preds.extend(spk_hat)
            emo_labels.extend(emo_gt)
            emo_preds.extend(emo_hat)
            total_samples += mels.size(0)

            loop.set_postfix({
                "mel": f"{np.mean(mel_losses):.3f}",
                "spk": f"{np.mean(spk_losses):.3f}",
                "emo": f"{np.mean(emo_losses):.3f}"
            })

        loop.close()

        if total_samples == 0:
            log_text += f"[{ckp_name}] - Похоже, тестовые данные пусты\n"
            continue

        mean_mel = float(np.mean(mel_losses))
        mean_spk = float(np.mean(spk_losses))
        mean_emo = float(np.mean(emo_losses))
        acc_spk = float(np.mean(np.array(spk_labels) == np.array(spk_preds)))
        acc_emo = float(np.mean(np.array(emo_labels) == np.array(emo_preds)))

        log_text += (f"=== Итог по {ckp_name} ===\n"
                     f"Сэмплов: {total_samples}\n"
                     f"MelLoss: {mean_mel:.4f}\n"
                     f"SpkLoss: {mean_spk:.4f}\n"
                     f"EmoLoss: {mean_emo:.4f}\n"
                     f"SpkAccuracy: {acc_spk:.4f}\n"
                     f"EmoAccuracy: {acc_emo:.4f}\n\n")

        spk_img = ""
        emo_img = ""

        if len(spk_map_) < 2000:
            cm_spk = confusion_matrix(spk_labels, spk_preds)
            if cm_spk.shape[0] <= 50:
                spk_img_b64 = create_confusion_image(
                    spk_labels,
                    spk_preds,
                    f"Speakers acc={acc_spk:.3f}",
                    cmap="Blues"
                )
                spk_img = f'<img src="data:image/png;base64,{spk_img_b64}" width="400px" style="border:1px solid #aaa"/>'
        if len(emo_map_) < 2000:
            cm_emo = confusion_matrix(emo_labels, emo_preds)
            if cm_emo.shape[0] <= 50:
                emo_img_b64 = create_confusion_image(
                    emo_labels,
                    emo_preds,
                    f"Emotions acc={acc_emo:.3f}",
                    cmap="Purples"
                )
                emo_img = f'<img src="data:image/png;base64,{emo_img_b64}" width="400px" style="border:1px solid #aaa"/>'

        if spk_img:
            spk_html_all += f"<h3>{ckp_name} - Speaker CM</h3>{spk_img}<br/>"
        else:
            spk_html_all += f"<h3>{ckp_name} - Speaker CM</h3>Нет матрицы<br/>"

        if emo_img:
            emo_html_all += f"<h3>{ckp_name} - Emotion CM</h3>{emo_img}<br/>"
        else:
            emo_html_all += f"<h3>{ckp_name} - Emotion CM</h3>Нет матрицы<br/>"

    return (log_text, spk_html_all, emo_html_all)

def launch_test_app():
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    all_pths = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.endswith(".pth")]
    all_pths.sort()

    with gr.Blocks() as demo:
        gr.Markdown("## Тестирование моделей на тестовом датасете")
        ckpt_multiselect = gr.CheckboxGroup(
            choices=all_pths,
            label="Выберите один или несколько чекпоинтов"
        )
        btn_eval = gr.Button("Запустить тест")
        txt_log = gr.Textbox(label="Лог процесса", lines=20)
        spk_html = gr.HTML(label="Speaker Confusion")
        emo_html = gr.HTML(label="Emotion Confusion")

        def evaluate_ckpts(ckpt_list):
            log_text, spk_html_content, emo_html_content = test_checkpoints(ckpt_list)
            return log_text, spk_html_content, emo_html_content

        btn_eval.click(
            fn=evaluate_ckpts,
            inputs=[ckpt_multiselect],
            outputs=[txt_log, spk_html, emo_html]
        )

        gr.Markdown("По завершении см. Confusion Matrix. Прогресс-бар виден в консоли.")

    demo.launch()

if __name__ == "__main__":
    launch_test_app()
