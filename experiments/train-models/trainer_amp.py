import os
import sys
import csv
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from configs import Config
from data import build_maps, create_loader
from models import VoiceCloningTTS
from losses import mel_reconstruction_loss, speaker_classification_loss, emotion_classification_loss
from text_preprocessing import normalize_text

torch.cuda.empty_cache()
cudnn.benchmark = True

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
    c2i = {c: i+1 for i, c in enumerate(chs)}
    return c2i

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

def load_if_exists(model, optimizer, scaler):
    cp = os.path.join(Config.CHECKPOINT_DIR, "last_checkpoint.pth")
    if os.path.isfile(cp):
        c = torch.load(cp, map_location=Config.DEVICE)
        model.load_state_dict(c["model"])
        optimizer.load_state_dict(c["optimizer"])
        scaler.load_state_dict(c["scaler"])
        return c["start_epoch"], c["global_step"], c["spk_map"], c["emo_map"], c["char_map"]
    return 0, 0, None, None, None

def save_checkpoint(model, optimizer, scaler, start_epoch, global_step, smap, emap, cmap, tag):
    cp = os.path.join(Config.CHECKPOINT_DIR, f"model_{tag}.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "start_epoch": start_epoch,
        "global_step": global_step,
        "spk_map": smap,
        "emo_map": emap,
        "char_map": cmap
    }, cp)

def train_loop_amp():
    print("Stage: Building speaker/emotion maps...")
    sys.stdout.flush()
    smap, emap = build_maps(Config.TRAIN_TSV)
    print(f"Speakers: {len(smap)}, Emotions: {len(emap)}")
    sys.stdout.flush()

    print("Stage: Building char-level vocab from TSV...")
    sys.stdout.flush()
    char_map = build_vocab_from_tsv(Config.TRAIN_TSV)
    print(f"Vocab size: {len(char_map)}")
    sys.stdout.flush()

    print("Stage: Creating DataLoader...")
    sys.stdout.flush()
    loader = create_loader(
        Config.TRAIN_TSV, 
        Config.TRAIN_WAVS, 
        smap, 
        emap, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True
    )
    n_batches = len(loader)
    print(f"Total batches: {n_batches}")
    sys.stdout.flush()

    print("Stage: Instantiating model on", Config.DEVICE)
    sys.stdout.flush()
    model = VoiceCloningTTS(
        vocab_size=len(char_map)+1,
        n_speakers=len(smap)+1,
        n_emotions=len(emap)+1
    ).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=Config.LOG_DIR)

    start_epoch, global_step, loaded_smap, loaded_emap, loaded_cmap = load_if_exists(model, optimizer, scaler)
    if loaded_smap is not None and loaded_emap is not None and loaded_cmap is not None:
        smap, emap, char_map = loaded_smap, loaded_emap, loaded_cmap
        print("Resumed from last_checkpoint.pth.")
    else:
        print("No existing checkpoint found, starting fresh.")

    print("Stage: Starting training loop with fp16 (mixed precision)...")
    sys.stdout.flush()

    for epoch in range(start_epoch, Config.EPOCHS):
        progress_bar = tqdm(loader, total=n_batches, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", ncols=120)
        quarter = n_batches // 4 if n_batches >= 4 else 1
        for i, batch in enumerate(progress_bar):
            mels = batch["mels"].to(Config.DEVICE, non_blocking=True)
            speakers = batch["speakers"].to(Config.DEVICE, non_blocking=True)
            emotions = batch["emotions"].to(Config.DEVICE, non_blocking=True)
            texts = batch["texts"]
            tokens = tokenize_batch(texts, char_map).to(Config.DEVICE, non_blocking=True)

            model.train()
            with torch.cuda.amp.autocast():
                mel_out, wav_out, spk_pred, emo_pred = model(mels, tokens)
                loss_mel = mel_reconstruction_loss(mel_out, mels)
                loss_spk = speaker_classification_loss(spk_pred, speakers)
                loss_emo = emotion_classification_loss(emo_pred, emotions)
                loss_total = loss_mel + loss_spk + loss_emo

            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("Loss/mel", loss_mel.item(), global_step)
            writer.add_scalar("Loss/spk", loss_spk.item(), global_step)
            writer.add_scalar("Loss/emo", loss_emo.item(), global_step)
            writer.add_scalar("Loss/total", loss_total.item(), global_step)

            progress_bar.set_postfix({
                "mel": f"{loss_mel.item():.3f}",
                "spk": f"{loss_spk.item():.3f}",
                "emo": f"{loss_emo.item():.3f}",
                "total": f"{loss_total.item():.3f}"
            })
            global_step += 1

            if (i+1) % quarter == 0 and (i+1) < n_batches:
                save_checkpoint(model, optimizer, scaler, epoch, global_step, smap, emap, char_map,
                                f"ep{epoch}_iter{i+1}")
        progress_bar.close()

        save_checkpoint(model, optimizer, scaler, epoch+1, global_step, smap, emap, char_map, f"ep{epoch}")
        torch.save({"model": model.state_dict()}, os.path.join(Config.CHECKPOINT_DIR, "last_model_only.pth"))
        save_checkpoint(model, optimizer, scaler, epoch+1, global_step, smap, emap, char_map, "last_checkpoint")
        print(f"Epoch {epoch+1} done -> model_ep{epoch}.pth (and last_checkpoint.pth)")
        sys.stdout.flush()

    writer.close()
    print("Training done.")
    sys.stdout.flush()

if __name__=="__main__":
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    train_loop_amp()
