import sys
import os
import json
import logging
import time
import requests
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from xttsv2.model import XTTS2Model
from xttsv2.data import text_to_token_ids, token_ids_to_text
from torch.utils.tensorboard import SummaryWriter
import jiwer  # NEW: для вычисления WER и CER

# Настройка логирования
logging.basicConfig(
    filename='training.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Telegram-уведомления
BOT_TOKEN = "your_bot_token_here"  # Заменить
CHAT_ID = "your_chat_id_here"      # Заменить

def send_telegram_message(text):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": text}
        requests.post(url, data=data)
        logging.info("Уведомление отправлено в Telegram.")
    except Exception as e:
        logging.warning(f"Ошибка Telegram: {e}")

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")
logging.info(f"Устройство: {device}")
torch.backends.cudnn.benchmark = True

# Датасет
class XTTS2Dataset(Dataset):
    def __init__(self, jsonl_file, features_dir):
        self.records = []
        self.features_dir = features_dir
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    path = os.path.join(features_dir, os.path.basename(record['audio_features']))
                    if os.path.exists(path):
                        self.records.append({'audio_features': path, 'text': record['text']})
                except Exception as e:
                    print(f"Ошибка в json: {e}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        try:
            r = self.records[idx]
            audio = np.load(r['audio_features'])
            text = r['text']
            return torch.tensor(audio, dtype=torch.float32), torch.tensor(text_to_token_ids(text), dtype=torch.long)
        except Exception as e:
            print(f"Ошибка в элементе {idx}: {e}")
            return torch.zeros(1), torch.zeros(1)

# Коллатор
def collate_fn(batch):
    af = [b[0] for b in batch]
    tt = [b[1] for b in batch]
    return pad_sequence(af, batch_first=True), pad_sequence(tt, batch_first=True)

# Конфиг
def save_training_config(cfg, path='training_config.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    logging.info(f"Сохранён конфиг: {path}")

# Параметры модели
def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Параметры: всего={total:,}, обучаемых={trainable:,}")
    print(f"Параметры модели: всего {total:,}, обучаемых {trainable:,}")

# Основная функция
def main():
    logging.info("Запуск обучения...")
    jsonl_file = 'D:/TrainerModel/Dataset/podcast_large.jsonl'
    features_dir = 'D:/TrainerModel/Dataset/features'
    checkpoint_path = 'D:/XTTS-v2/xtts2_finetuned.pth'
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_checkpoint_path = 'D:/XTTS-v2/model.pth'
    speakers_checkpoint_path = 'D:/XTTS-v2/speakers_xtts.pth'
    batch_size = 32
    validation_split = 0.2
    num_epochs = 20
    learning_rate = 1e-4

    cfg = {
        "jsonl_file": jsonl_file,
        "features_dir": features_dir,
        "checkpoint_path": checkpoint_path,
        "model_checkpoint_path": model_checkpoint_path,
        "speakers_checkpoint_path": speakers_checkpoint_path,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "device": str(device)
    }
    save_training_config(cfg)

    dataset = XTTS2Dataset(jsonl_file, features_dir)
    val_len = int(validation_split * len(dataset))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = XTTS2Model.from_pretrained(model_checkpoint_path, speaker_checkpoint=speakers_checkpoint_path)
    model.to(device).half()
    count_model_params(model)

    opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir='runs/xtts2_training')

    pred_path = "predictions.txt"
    if os.path.exists(pred_path):
        os.remove(pred_path)

    best_val = float('inf')
    train_losses, val_losses = [], []
    patience, no_improve = 5, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for af, tt in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]"):
            af, tt = af.to(device), tt.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(af, tt)
                loss = loss_fn(out.view(-1, out.size(-1)), tt.view(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        wer_total, cer_total, samples = 0.0, 0.0, 0

        with torch.no_grad():
            for af, tt in val_loader:
                af, tt = af.to(device), tt.to(device)
                with torch.cuda.amp.autocast():
                    out = model(af, tt)
                    loss = loss_fn(out.view(-1, out.size(-1)), tt.view(-1))
                val_loss += loss.item()

                for i in range(min(af.size(0), 3)):
                    true = token_ids_to_text(tt[i].cpu().numpy())
                    pred = token_ids_to_text(out[i].argmax(-1).cpu().numpy())
                    with open(pred_path, "a", encoding="utf-8") as f:
                        f.write(f"[Epoch {epoch+1}] Истинный текст: {true}\n")
                        f.write(f"[Epoch {epoch+1}] Предсказание  : {pred}\n\n")
                    wer_total += jiwer.wer(true, pred)
                    cer_total += jiwer.cer(true, pred)
                    samples += 1

                del out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        avg_train = total_loss / len(train_loader)
        avg_wer = wer_total / samples if samples else 0.0
        avg_cer = cer_total / samples if samples else 0.0

        print(f"Эпоха {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={val_loss:.4f}, WER={avg_wer:.2%}, CER={avg_cer:.2%}")
        logging.info(f"Train Loss={avg_train:.4f}, Val Loss={val_loss:.4f}, WER={avg_wer:.2%}, CER={avg_cer:.2%}")

        writer.add_scalar("Loss/Train", avg_train, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("WER", avg_wer, epoch + 1)
        writer.add_scalar("CER", avg_cer, epoch + 1)

        train_losses.append(avg_train)
        val_losses.append(val_loss)
        sched.step()

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'val_loss': val_loss
        }, os.path.join(checkpoints_dir, f"epoch_{epoch+1:03}.pth"))

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            logging.info("Сохранена лучшая модель.")
            send_telegram_message(f"✅ Эпоха {epoch+1}: новая лучшая модель!\nVal Loss: {val_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info("Ранняя остановка.")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logging.info("Очищена CUDA-память после эпохи.")

    writer.close()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")
    logging.info("Сохранён график потерь.")
    send_telegram_message(f"🛑 Обучение завершено. Последняя эпоха: {epoch+1}\nVal Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
