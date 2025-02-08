import os
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from configs import Config
from data import build_maps, create_loader
from models import VoiceCloningTTS
from losses import mel_reconstruction_loss, speaker_classification_loss, emotion_classification_loss
from text_preprocessing import normalize_text
from torch.utils.tensorboard import SummaryWriter
import itertools

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
    mx = max(len(x) for x in txts) if txts else 1
    out = []
    for t in txts:
        arr = []
        for ch in t:
            arr.append(c2i[ch] if ch in c2i else 0)
        while len(arr) < mx:
            arr.append(0)
        out.append(arr)
    return torch.LongTensor(out)

def load_checkpoint(ckpt_path):
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

def evaluate_model(test_tsv, test_wavs, ckpt_path, out_dir="./eval"):
    os.makedirs(out_dir, exist_ok=True)
    model, smap, emap, cmap = load_checkpoint(ckpt_path)
    test_loader = create_loader(test_tsv, test_wavs, smap, emap, batch_size=Config.BATCH_SIZE, shuffle=False)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb_logs"))

    spk_labels = []
    spk_preds = []
    emo_labels = []
    emo_preds = []
    mel_losses = []
    spk_losses = []
    emo_losses = []

    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            mels = batch["mels"].to(Config.DEVICE)
            spk = batch["speakers"].to(Config.DEVICE)
            emo = batch["emotions"].to(Config.DEVICE)
            txts = batch["texts"]
            tokens = tokenize_batch(txts, cmap).to(Config.DEVICE)

            mel_out, wav_out, spk_pred, emo_pred = model(mels, tokens)
            l_mel = mel_reconstruction_loss(mel_out, mels).item()
            l_spk = speaker_classification_loss(spk_pred, spk).item()
            l_emo = emotion_classification_loss(emo_pred, emo).item()

            mel_losses.append(l_mel)
            spk_losses.append(l_spk)
            emo_losses.append(l_emo)

            spk_hat = torch.argmax(spk_pred, dim=1).cpu().numpy()
            emo_hat = torch.argmax(emo_pred, dim=1).cpu().numpy()
            spk_gt = spk.cpu().numpy()
            emo_gt = emo.cpu().numpy()

            spk_labels.extend(spk_gt)
            spk_preds.extend(spk_hat)
            emo_labels.extend(emo_gt)
            emo_preds.extend(emo_hat)
            total_samples += mels.size(0)

    mean_mel = np.mean(mel_losses)
    mean_spk = np.mean(spk_losses)
    mean_emo = np.mean(emo_losses)

    acc_spk = np.mean(np.array(spk_labels) == np.array(spk_preds)) if total_samples>0 else 0
    acc_emo = np.mean(np.array(emo_labels) == np.array(emo_preds)) if total_samples>0 else 0

    writer.add_scalar("Eval/MelLoss", mean_mel, 0)
    writer.add_scalar("Eval/SpkLoss", mean_spk, 0)
    writer.add_scalar("Eval/EmoLoss", mean_emo, 0)
    writer.add_scalar("Eval/SpkAccuracy", acc_spk, 0)
    writer.add_scalar("Eval/EmoAccuracy", acc_emo, 0)
    writer.close()

    cm_spk = confusion_matrix(spk_labels, spk_preds)
    cm_emo = confusion_matrix(emo_labels, emo_preds)
    spk_png = os.path.join(out_dir, "confusion_speakers.png")
    emo_png = os.path.join(out_dir, "confusion_emotions.png")
    if cm_spk.shape[0] <= 50:
        disp_spk = ConfusionMatrixDisplay(cm_spk)
        disp_spk.plot()
        plt.title(f"Speaker Confusion Matrix (Acc={acc_spk:.3f})")
        plt.savefig(spk_png)
        plt.close()
    if cm_emo.shape[0] <= 50:
        disp_emo = ConfusionMatrixDisplay(cm_emo)
        disp_emo.plot()
        plt.title(f"Emotion Confusion Matrix (Acc={acc_emo:.3f})")
        plt.savefig(emo_png)
        plt.close()

    rep_txt = os.path.join(out_dir, "test_report.txt")
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Mean Mel Loss: {mean_mel:.4f}\n")
        f.write(f"Mean Speaker Loss: {mean_spk:.4f}\n")
        f.write(f"Mean Emotion Loss: {mean_emo:.4f}\n")
        f.write(f"Speaker Accuracy: {acc_spk:.4f}\n")
        f.write(f"Emotion Accuracy: {acc_emo:.4f}\n")
        f.write("Confusion Matrices saved as images.\n")

def run_test():
    ckpt = os.path.join(Config.CHECKPOINT_DIR, "model_ep0.pth")  # пример
    evaluate_model(Config.TEST_TSV, Config.TEST_WAVS, ckpt)

if __name__=="__main__":
    run_test()
