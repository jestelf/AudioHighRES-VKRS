# trainer_big_debug_fix.py
import os
import csv
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils as nnutils
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from configs import Config
from data import build_maps, TTSDataset, collate_fn
from models_big import BigModel
from losses import mel_reconstruction_loss, speaker_classification_loss, emotion_classification_loss
from text_preprocessing import normalize_text

torch.cuda.empty_cache()
cudnn.benchmark = True

def build_vocab_limited(tsv_file, limit=10000):
    txt_rows = []
    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= limit:
                break
            st = row.get("speaker_text", "")
            txt_rows.append(normalize_text(st))
    chars = set()
    for txt in txt_rows:
        for c in txt:
            chars.add(c)
    return {c: i+1 for i, c in enumerate(sorted(list(chars)))}

def create_small_dataset(tsv_file, wav_dir, sp_map, em_map, max_records=10000):
    ds = TTSDataset(tsv_file, wav_dir, sp_map, em_map, max_dur=15.0, min_dur=0.2)
    if len(ds) > max_records:
        ds = Subset(ds, list(range(max_records)))
    return ds

def tokenize_batch(txts, c2i):
    if not txts:
        return torch.LongTensor([[]])
    mx = 1
    for t in txts:
        if len(t) > mx:
            mx = len(t)
    out = []
    for t in txts:
        arr = []
        for ch in t:
            arr.append(c2i[ch] if ch in c2i else 0)
        if len(arr)==0:
            arr=[0]
        while len(arr)<mx:
            arr.append(0)
        out.append(arr)
    return torch.LongTensor(out)

def train_big_debug_fix():
    sp_map, em_map = build_maps(Config.TRAIN_TSV)
    c_map = build_vocab_limited(Config.TRAIN_TSV, limit=10000)
    ds_small = create_small_dataset(Config.TRAIN_TSV, Config.TRAIN_WAVS, sp_map, em_map, max_records=10000)
    dl_small = DataLoader(ds_small, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    model = BigModel(len(c_map)+1, len(sp_map)+1, len(em_map)+1).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, betas=(0.9,0.99))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=1)
    writer = SummaryWriter(Config.LOG_DIR)
    scaler = torch.amp.GradScaler()
    global_step=0

    for ep in range(Config.EPOCHS):
        loop = tqdm(dl_small, desc=f"[BigDebug Epoch {ep+1}/{Config.EPOCHS}]", ncols=120)
        mel_list, spk_list, emo_list = [], [], []
        for i, batch in enumerate(loop):
            if batch is None:
                continue
            mels = batch["mels"].to(Config.DEVICE, non_blocking=True)
            sps = batch["speakers"].to(Config.DEVICE, non_blocking=True)
            ems = batch["emotions"].to(Config.DEVICE, non_blocking=True)
            txts = batch["texts"]
            toks = tokenize_batch(txts, c_map).to(Config.DEVICE)
            model.train()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                mel_out, wav_out, sp_logits, emo_logits = model(mels, toks)
                loss_mel = mel_reconstruction_loss(mel_out, mels)
                loss_spk = speaker_classification_loss(sp_logits, sps)
                loss_emo = emotion_classification_loss(emo_logits, ems)
                loss_total = loss_mel + loss_spk + loss_emo

            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            nnutils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            mel_list.append(loss_mel.item())
            spk_list.append(loss_spk.item())
            emo_list.append(loss_emo.item())
            writer.add_scalar("BigDebug/Mel", loss_mel.item(), global_step)
            writer.add_scalar("BigDebug/Spk", loss_spk.item(), global_step)
            writer.add_scalar("BigDebug/Emo", loss_emo.item(), global_step)
            writer.add_scalar("BigDebug/Total", loss_total.item(), global_step)
            loop.set_postfix({
                "mel":f"{loss_mel.item():.3f}",
                "spk":f"{loss_spk.item():.3f}",
                "emo":f"{loss_emo.item():.3f}"
            })
            global_step+=1
        mmean = sum(mel_list)/len(mel_list) if len(mel_list)>0 else 0
        smean = sum(spk_list)/len(spk_list) if len(spk_list)>0 else 0
        emean = sum(emo_list)/len(emo_list) if len(emo_list)>0 else 0
        scheduler.step(mmean+smean+emean)
        ckpt = {
            "model":model.state_dict(),
            "spk_map":sp_map,
            "emo_map":em_map,
            "char_map":c_map
        }
        torch.save(ckpt, os.path.join(Config.CHECKPOINT_DIR, f"bigdebug_ep{ep}.pth"))
    writer.close()

if __name__=="__main__":
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    train_big_debug_fix()
