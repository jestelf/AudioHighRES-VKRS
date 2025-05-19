import sys
import os
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt  # NEW
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from xttsv2.model import XTTS2Model
from xttsv2.data import text_to_token_ids

# (логгирование и device — без изменений)

# Основная функция
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    train_losses = []  # NEW
    val_losses = []    # NEW

    for epoch in range(num_epochs):
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

        val_loss /= len(val_loader)
        train_loss_avg = running_loss / len(train_loader)
        print(f"Эпоха {epoch + 1}: Обучение Loss={train_loss_avg:.4f} | Валидация Loss={val_loss:.4f}")
        logging.info(f"Эпоха {epoch + 1}: Train Loss={train_loss_avg:.4f} | Val Loss={val_loss:.4f}")

        train_losses.append(train_loss_avg)  # NEW
        val_losses.append(val_loss)         # NEW

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Модель сохранена: {checkpoint_path}")
            logging.info(f"Лучшая модель сохранена на эпохе {epoch + 1} с val_loss={val_loss:.4f}")

    # Визуализация потерь # NEW
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
    logging.info("График потерь сохранён в loss_plot.png")  # NEW

    print("Обучение завершено.")
    logging.info("Обучение завершено.")

if __name__ == "__main__":
    main()
