# test_debug.py
import os
import torch
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from configs import Config
from data import build_maps, TTSDataset, collate_fn
from models import VoiceCloningTTS
from text_preprocessing import normalize_text
from losses import mel_reconstruction_loss, speaker_classification_loss, emotion_classification_loss

def load_checkpoint(ckpt_path):
    c = torch.load(ckpt_path, map_location=Config.DEVICE)
    m = VoiceCloningTTS(
        vocab_size=len(c["char_map"])+1,
        n_spk=len(c["spk_map"])+1,
        n_emo=len(c["emo_map"])+1
    )
    m.load_state_dict(c["model"])
    m.eval().to(Config.DEVICE)
    return m, c["spk_map"], c["emo_map"], c["char_map"]

def build_test_loader(tsv_file, wav_dir, sp_map, em_map, max_samples=100):
    ds = TTSDataset(tsv_file, wav_dir, sp_map, em_map, max_dur=15.0, min_dur=0.2)
    if len(ds)>max_samples:
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))
    dl = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return dl

def evaluate_checkpoint(ckpt_path, test_tsv, test_wavs, max_samples=100):
    if not os.path.isfile(ckpt_path):
        return f"Checkpoint {ckpt_path} not found", None, None
    model, smap, emap, cmap = load_checkpoint(ckpt_path)
    test_loader = build_test_loader(test_tsv, test_wavs, smap, emap, max_samples=max_samples)
    spk_preds, spk_true = [], []
    emo_preds, emo_true = [], []
    mel_losses, spk_losses, emo_losses = [], [], []
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {os.path.basename(ckpt_path)}"):
            if batch is None:
                continue
            mels = batch["mels"].to(Config.DEVICE)
            sps = batch["speakers"].to(Config.DEVICE)
            ems = batch["emotions"].to(Config.DEVICE)
            txts = batch["texts"]
            # В этом тестовом коде не обязательно нужно токенизировать заново,
            # т.к. мы не восстанавливаем текст, а просто считаем лосс. Но для честности:
            tok = []
            for t in txts:
                arr = []
                for c in normalize_text(t):
                    arr.append(cmap[c] if c in cmap else 0)
                if not arr:
                    arr=[0]
                tok.append(arr)
            mx = max(len(a) for a in tok)
            tok_ = []
            for a in tok:
                while len(a)<mx: a.append(0)
                tok_.append(a)
            tok_t = torch.LongTensor(tok_).to(Config.DEVICE)
            mo, wo, sp, eo = model(mels, tok_t)
            lm = mel_reconstruction_loss(mo, mels).item()
            ls = speaker_classification_loss(sp, sps).item()
            le = emotion_classification_loss(eo, ems).item()
            mel_losses.append(lm)
            spk_losses.append(ls)
            emo_losses.append(le)
            spk_hat = torch.argmax(sp, dim=1).cpu().numpy()
            emo_hat = torch.argmax(eo, dim=1).cpu().numpy()
            spk_gt = sps.cpu().numpy()
            emo_gt = ems.cpu().numpy()
            spk_preds.extend(spk_hat)
            spk_true.extend(spk_gt)
            emo_preds.extend(emo_hat)
            emo_true.extend(emo_gt)
            total+=mels.size(0)
    if total==0:
        return "No samples in test dataset", None, None
    mm = float(np.mean(mel_losses))
    ms = float(np.mean(spk_losses))
    me = float(np.mean(emo_losses))
    spk_acc = float((np.array(spk_preds)==np.array(spk_true)).mean())
    emo_acc = float((np.array(emo_preds)==np.array(emo_true)).mean())
    txt_rep = (f"Checkpoint: {ckpt_path}\n"
               f"Samples tested: {total}\n"
               f"MelLoss: {mm:.4f}\n"
               f"SpkLoss: {ms:.4f}\n"
               f"EmoLoss: {me:.4f}\n"
               f"SpkAcc: {spk_acc:.4f}\n"
               f"EmoAcc: {emo_acc:.4f}\n")
    cm_spk, cm_emo = None, None
    if len(smap)<1000:
        cm_spk = confusion_matrix(spk_true, spk_preds)
    if len(emap)<1000:
        cm_emo = confusion_matrix(emo_true, emo_preds)
    return txt_rep, cm_spk, cm_emo

def save_cm_plot(cm, outpng, title="Confusion", cmap="Blues"):
    fig, ax = plt.subplots(figsize=(5,5))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title)
    fig.savefig(outpng, bbox_inches="tight", dpi=100)
    plt.close(fig)

def main():
    ckpt_dir = Config.CHECKPOINT_DIR
    cands = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    cands.sort()
    for c in cands:
        ck = os.path.join(ckpt_dir, c)
        rep, cspk, cemo = evaluate_checkpoint(
            ck, 
            Config.TEST_TSV, 
            Config.TEST_WAVS,
            max_samples=100
        )
        print(rep)
        if cspk is not None and cspk.shape[0]<=50:
            outp = os.path.join(ckpt_dir, f"{c}_spkCM.png")
            save_cm_plot(cspk, outp, title=f"SpkCM {c}", cmap="Blues")
        if cemo is not None and cemo.shape[0]<=50:
            outp = os.path.join(ckpt_dir, f"{c}_emoCM.png")
            save_cm_plot(cemo, outp, title=f"EmoCM {c}", cmap="Purples")

if __name__=="__main__":
    main()
