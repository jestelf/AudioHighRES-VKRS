# inference.py
import os
import torch
import torchaudio
import numpy as np
from configs import Config
from models import VoiceCloningTTS
from text_preprocessing import normalize_text

def load_model(ckpt_path):
    c = torch.load(ckpt_path, map_location=Config.DEVICE)
    model = VoiceCloningTTS(len(c["char_map"])+1, len(c["spk_map"])+1, len(c["emo_map"])+1)
    model.load_state_dict(c["model"])
    model.eval().to(Config.DEVICE)
    return model, c["char_map"]

def text2tokens(txt, cmap):
    arr = []
    for ch in txt:
        arr.append(cmap[ch] if ch in cmap else 0)
    if not arr:
        arr=[0]
    return torch.LongTensor(arr).unsqueeze(0)

def infer_tts(ckpt, txt, ref_wav, outwav="synth.wav"):
    model, cm = load_model(ckpt)
    w, sr = torchaudio.load(ref_wav)
    if sr!=Config.SAMPLE_RATE:
        w = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)(w)
    w = w.mean(dim=0, keepdim=True)
    mel_ex = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_mels=Config.N_MEL_CHANNELS,
        n_fft=Config.N_FFT,
        win_length=Config.WIN_LENGTH,
        hop_length=Config.HOP_LENGTH
    )
    rm = mel_ex(w)
    nt = normalize_text(txt)
    tk = text2tokens(nt, cm).to(Config.DEVICE)
    rm = rm.unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        mo, wo, _, _ = model(rm, tk)
    au = wo.squeeze(0).cpu().unsqueeze(0)
    torchaudio.save(outwav, au, Config.SAMPLE_RATE)

if __name__=="__main__":
    cp = os.path.join(Config.CHECKPOINT_DIR, "model_ep0.pth")
    txt = "Привет, как твои дела"
    ref = r"D:\DatasetDusha\crowd_test\wavs\audio.wav"
    infer_tts(cp, txt, ref, "syn_out.wav")
