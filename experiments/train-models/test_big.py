# test_big_debug.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configs import Config
from data import TTSDataset, build_maps, collate_fn
from models_big import BigModel
from text_preprocessing import normalize_text
from losses import mel_reconstruction_loss, speaker_classification_loss, emotion_classification_loss

def load_big_ckpt(ckpt):
    st = torch.load(ckpt, map_location=Config.DEVICE)
    model = BigModel(len(st["char_map"])+1, len(st["spk_map"])+1, len(st["emo_map"])+1)
    model.load_state_dict(st["model"])
    model.eval().to(Config.DEVICE)
    return model, st["spk_map"], st["emo_map"], st["char_map"]

def token_batch(txts, c2i):
    if not txts:
        return torch.LongTensor([[]])
    mx = max(len(x) for x in txts)
    out=[]
    for t in txts:
        arr=[]
        for c in t:
            arr.append(c2i[c] if c in c2i else 0)
        while len(arr)<mx:
            arr.append(0)
        out.append(arr)
    return torch.LongTensor(out)

def test_big_debug(ckpt, test_tsv, test_wavs, max_samples=100):
    if not os.path.isfile(ckpt):
        return None
    m, sm, em, cm = load_big_ckpt(ckpt)
    ds = TTSDataset(test_tsv, test_wavs, sm, em, 15.0, 0.2)
    if len(ds)>max_samples:
        ds = Subset(ds, list(range(max_samples)))
    dl = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    spk_pred, spk_true = [], []
    emo_pred, emo_true = [], []
    mel_ls, spk_ls, emo_ls = [], [], []
    count=0
    with torch.no_grad():
        for b in tqdm(dl, desc=f"Test {os.path.basename(ckpt)}"):
            ms=b["mels"].to(Config.DEVICE)
            ss=b["speakers"].to(Config.DEVICE)
            es=b["emotions"].to(Config.DEVICE)
            tx=b["texts"]
            tk=token_batch(tx,cm).to(Config.DEVICE)
            mo,wo,sp,eo=m(ms,tk)
            lm=mel_reconstruction_loss(mo, ms).item()
            ls=speaker_classification_loss(sp, ss).item()
            le=emotion_classification_loss(eo, es).item()
            mel_ls.append(lm)
            spk_ls.append(ls)
            emo_ls.append(le)
            s_hat=sp.argmax(dim=1).cpu().numpy()
            e_hat=eo.argmax(dim=1).cpu().numpy()
            spk_pred.extend(s_hat)
            spk_true.extend(ss.cpu().numpy())
            emo_pred.extend(e_hat)
            emo_true.extend(es.cpu().numpy())
            count+=ms.size(0)
    if count==0:
        return "No test samples"
    mm=float(np.mean(mel_ls))
    ms=float(np.mean(spk_ls))
    me=float(np.mean(emo_ls))
    sa=float((np.array(spk_pred)==np.array(spk_true)).mean())
    ea=float((np.array(emo_pred)==np.array(emo_true)).mean())
    return f"{ckpt}\nNumSamples:{count}\nMel:{mm:.3f}\nSpk:{ms:.3f}\nEmo:{me:.3f}\nSpkAcc:{sa:.3f}\nEmoAcc:{ea:.3f}\n"

def main():
    cpts=[f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith("bigdebug_ep") and f.endswith(".pth")]
    cpts.sort()
    for c in cpts:
        path=os.path.join(Config.CHECKPOINT_DIR,c)
        rep=test_big_debug(path,Config.TEST_TSV,Config.TEST_WAVS,100)
        print(rep)

if __name__=="__main__":
    main()
