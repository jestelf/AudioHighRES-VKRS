# data.py
import os
import csv
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from configs import Config
from text_preprocessing import normalize_text

class TTSDataset(Dataset):
    def __init__(self, tsv_file, wav_dir, spk_map, emo_map, max_dur=15.0, min_dur=0.2):
        self.entries = []
        with open(tsv_file, "r", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                ap = row.get("audio_path", "")
                if not ap:
                    continue
                st = row.get("speaker_text", "")
                if not st:
                    st = " "
                ds = row.get("duration", "")
                if not ds:
                    continue
                try:
                    dur = float(ds)
                except:
                    continue
                if dur < min_dur or dur > max_dur:
                    continue
                emo = row.get("annotator_emo", "neutral")
                if not emo:
                    emo = "neutral"
                spk = row.get("annotator_id", "unknown")
                if not spk:
                    spk = "unknown"
                self.entries.append((ap, st, emo, spk))
        self.wav_dir = wav_dir
        self.spk_map = spk_map
        self.emo_map = emo_map
        self.mel_ex = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_mels=Config.N_MEL_CHANNELS,
            n_fft=Config.N_FFT,
            win_length=Config.WIN_LENGTH,
            hop_length=Config.HOP_LENGTH
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        p, txt, emo, spk = self.entries[i]
        fpath = os.path.join(self.wav_dir, os.path.basename(p))
        wav, sr = torchaudio.load(fpath)
        if sr != Config.SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)(wav)
        wav = wav.mean(dim=0, keepdim=True)
        mel = self.mel_ex(wav)
        text = normalize_text(txt)
        si = 0
        if spk in self.spk_map:
            si = self.spk_map[spk]
        ei = 0
        if emo in self.emo_map:
            ei = self.emo_map[emo]
        return mel, text, si, ei

def collate_fn(batch):
    ms, ts, ss, es = zip(*batch)
    ln = [m.shape[2] for m in ms]
    mx = max(ln)
    mp = torch.zeros(len(ms), Config.N_MEL_CHANNELS, mx)
    for i, m in enumerate(ms):
        mp[i, :, :m.shape[2]] = m[0, :, :]
    return {"mels": mp, "texts": list(ts), "speakers": torch.LongTensor(ss), "emotions": torch.LongTensor(es)}

def build_maps(tsv_file):
    spk_s, emo_s = set(), set()
    with open(tsv_file, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            spk = row.get("annotator_id", "unknown")
            if not spk:
                spk = "unknown"
            emo = row.get("annotator_emo", "neutral")
            if not emo:
                emo = "neutral"
            spk_s.add(spk)
            emo_s.add(emo)
    s_l = sorted(list(spk_s))
    e_l = sorted(list(emo_s))
    smap = {s: i + 1 for i, s in enumerate(s_l)}
    emap = {e: i + 1 for i, e in enumerate(e_l)}
    return smap, emap

def create_loader(tsv_file, wav_dir, smap, emap, bs=Config.BATCH_SIZE, sh=True):
    ds = TTSDataset(tsv_file, wav_dir, smap, emap)
    return DataLoader(ds, batch_size=bs, shuffle=sh, pin_memory=True, collate_fn=collate_fn)
