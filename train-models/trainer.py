# trainer.py
import os
import sys
import csv
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils as nnutils
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from configs import Config
from data import build_maps, create_loader
from models import VoiceCloningTTS
from losses import mel_reconstruction_loss, speaker_classification_loss, emotion_classification_loss
from text_preprocessing import normalize_text

torch.cuda.empty_cache()
cudnn.benchmark = True

def build_vocab(txtfile):
    texts = []
    with open(txtfile, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            st = row.get("speaker_text", "")
            texts.append(normalize_text(st))
    chars = set()
    for t in texts:
        for c in t:
            chars.add(c)
    chars = sorted(list(chars))
    return {c: i+1 for i, c in enumerate(chars)}

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

def train_loop():
    smap, emap = build_maps(Config.TRAIN_TSV)
    char_map = build_vocab(Config.TRAIN_TSV)
    loader = create_loader(
        Config.TRAIN_TSV,
        Config.TRAIN_WAVS,
        smap,
        emap,
        bs=Config.BATCH_SIZE,
        sh=True
    )
    model = VoiceCloningTTS(
        vocab_size=len(char_map)+1,
        n_spk=len(smap)+1,
        n_emo=len(emap)+1
    ).to(Config.DEVICE)

    opt = optim.Adam(model.parameters(), lr=Config.LR)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, verbose=True)
    writer = SummaryWriter(Config.LOG_DIR)
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    for epoch in range(Config.EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", ncols=120)
        mel_losses, spk_losses, emo_losses = [], [], []
        for batch_idx, batch in enumerate(loop):
            if batch is None:
                continue
            mels = batch["mels"].to(Config.DEVICE, non_blocking=True)
            speakers = batch["speakers"].to(Config.DEVICE, non_blocking=True)
            emotions = batch["emotions"].to(Config.DEVICE, non_blocking=True)
            texts = batch["texts"]
            tokens = tokenize_batch(texts, char_map).to(Config.DEVICE, non_blocking=True)

            model.train()
            with torch.cuda.amp.autocast():
                mel_out, wav_out, spk_logits, emo_logits = model(mels, tokens)
                loss_mel = mel_reconstruction_loss(mel_out, mels)
                loss_spk = speaker_classification_loss(spk_logits, speakers)
                loss_emo = emotion_classification_loss(emo_logits, emotions)
                loss_total = loss_mel + loss_spk + loss_emo

            opt.zero_grad()
            scaler.scale(loss_total).backward()
            nnutils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            mel_losses.append(loss_mel.item())
            spk_losses.append(loss_spk.item())
            emo_losses.append(loss_emo.item())

            writer.add_scalar("Loss/Mel", loss_mel.item(), global_step)
            writer.add_scalar("Loss/Spk", loss_spk.item(), global_step)
            writer.add_scalar("Loss/Emo", loss_emo.item(), global_step)
            writer.add_scalar("Loss/Total", loss_total.item(), global_step)

            loop.set_postfix({
                "mel": f"{loss_mel.item():.3f}",
                "spk": f"{loss_spk.item():.3f}",
                "emo": f"{loss_emo.item():.3f}"
            })
            global_step += 1

        mean_mel = sum(mel_losses)/len(mel_losses) if len(mel_losses)>0 else 0
        mean_spk = sum(spk_losses)/len(spk_losses) if len(spk_losses)>0 else 0
        mean_emo = sum(emo_losses)/len(emo_losses) if len(emo_losses)>0 else 0
        sum_loss = mean_mel + mean_spk + mean_emo
        sched.step(sum_loss)
        ckpt = {
            "model": model.state_dict(),
            "spk_map": smap,
            "emo_map": emap,
            "char_map": char_map
        }
        torch.save(ckpt, os.path.join(Config.CHECKPOINT_DIR, f"model_ep{epoch}.pth"))
    writer.close()

if __name__=="__main__":
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    train_loop()
