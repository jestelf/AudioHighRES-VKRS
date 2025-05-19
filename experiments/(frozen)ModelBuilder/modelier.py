import sys
import os
import json
import logging
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from xttsv2.model import XTTS2Model
from xttsv2.data import text_to_token_ids, token_ids_to_text  # NEW
from torch.utils.tensorboard import SummaryWriter  # NEW

# Настройка логирования
logging.basicConfig(
    filename='training.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")
logging.info(f"Используется устройство: {device}")
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
                    audio_path = os.path.join(features_dir, os.path.basename(record['audio_features']))
                    if os.path.exists(audio_path):
                        self.records.append({
                            'audio_features': audio_path,
                            'text': record['text']
                        })
                except Exception as e:
                    print(f"Ошибка при загрузке записи: {e}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        try:
            record = self.records[idx]
            audio_features = np.load(record['audio_features'])
            text = record['text']

            audio_features = torch.tensor(audio_features, dtype=torch.float32)
            text_tokens = torch.tensor(text_to_token_ids(text), dtype=torch.long)
            return audio_features, text_tokens
        except Exception as e:
            print(f"Ошибка в записи {idx}: {e}")
            return torch.zeros(1), torch.zeros(1)

# Батч
def collate_fn(batch):
    audio_features = [item[0] for item in batch]
    text_tokens = [item[1] for item in batch]
    audio_features = pad_sequence(audio_features, batch_first=True, padding_value=0)
    text_tokens = pad_sequence(text_tokens, batch_first=True, padding_value=0)
    return audio_features, text_tokens

# Сохраняем конфиг обучения
def save_training_config(config_dict, path='training_config.json'):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        logging.info(f"Конфигурация обучения сохранена в {path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении конфигурации: {e}")

# Параметры модели
def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Всего параметров в модели: {total:,}")
    logging.info(f"Обучаемых параметров: {trainable:,}")
    print(f"Параметры модели: всего {total:,}, обучаемых {trainable:,}")

# Основной запуск
def main():
    logging.info("Инициализация обучения...")

    jsonl_file = 'D:/TrainerModel/Dataset/podcast_large.jsonl'
    features_dir = 'D:/TrainerModel/Dataset/features'
    checkpoint_path = 'D:/XTTS-v2/xtts2_finetuned.pth'
    model_checkpoint_path = 'D:/XTTS-v2/model.pth'
    speakers_checkpoint_path = 'D:/XTTS-v2/speakers_xtts.pth'
    batch_size = 32
    validation_split = 0.2
    num_epochs = 20
    learning_rate = 1e-4

    config = {
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
    save_training_config(config)

    dataset = XTTS2Dataset(jsonl_file, features_dir)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("Загрузка модели XTTS2...")
    logging.info("Загрузка модели XTTS2...")
    model = XTTS2Model.from_pretrained(model_checkpoint_path, speaker_checkpoint=speakers_checkpoint_path)
    model.to(device).half()
    count_model_params(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir='runs/xtts2_training')
    predictions_log_path = "predictions.txt"
    if os.path.exists(predictions_log_path):
        os.remove(predictions_log_path)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping_patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        for audio_features, text_tokens in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]"):
            audio_features, text_tokens = audio_features.to(device), text_tokens.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(audio_features, text_tokens)
                loss = criterion(outputs.view(-1, outputs.size(-1)), text_tokens.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio_features, text_tokens in val_loader:
                audio_features, text_tokens = audio_features.to(device), text_tokens.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(audio_features, text_tokens)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), text_tokens.view(-1))
                val_loss += loss.item()

                # Прогнозы — логируем первые 3
                with open(predictions_log_path, "a", encoding="utf-8") as pred_f:
                    for i in range(min(3, audio_features.size(0))):
                        true_tokens = text_tokens[i].detach().cpu().numpy()
                        pred_tokens = outputs[i].argmax(dim=-1).detach().cpu().numpy()
                        true_text = token_ids_to_text(true_tokens)
                        pred_text = token_ids_to_text(pred_tokens)
                        pred_f.write(f"[Epoch {epoch+1}] Истинный текст: {true_text}\n")
                        pred_f.write(f"[Epoch {epoch+1}] Предсказание  : {pred_text}\n\n")

        val_loss /= len(val_loader)
        train_loss_avg = running_loss / len(train_loader)

        print(f"Эпоха {epoch + 1}: Обучение Loss={train_loss_avg:.4f} | Валидация Loss={val_loss:.4f}")
        logging.info(f"Эпоха {epoch + 1}: Train Loss={train_loss_avg:.4f} | Val Loss={val_loss:.4f}")
        logging.info(f"Время эпохи {epoch + 1}: {time.time() - epoch_start_time:.2f} сек.")

        writer.add_scalar('Loss/Train', train_loss_avg, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Модель сохранена: {checkpoint_path}")
            logging.info(f"Лучшая модель сохранена на эпохе {epoch + 1} с val_loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"Нет улучшения ({epochs_no_improve}/{early_stopping_patience})")
            if epochs_no_improve >= early_stopping_patience:
                print(f"Ранняя остановка на эпохе {epoch + 1}")
                logging.info("Ранняя остановка обучения из-за отсутствия улучшений.")
                break

    writer.close()

    # Визуализация потерь
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    logging.info("График потерь сохранён в loss_plot.png")

    print("Обучение завершено.")
    logging.info("Обучение завершено.")

if __name__ == "__main__":
    main()
