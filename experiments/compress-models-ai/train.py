# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
from tqdm import tqdm
from model import VQVAE
from dataset import CommonVoiceRussianDataset

def setup_logging(log_file="training.log"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_file, mode='w')])
    logging.info("Logging is set up. Logs will be saved in '%s'", log_file)

def collate_fn(batch):
    """
    Функция для объединения элементов батча.
    Каждый элемент батча — кортеж (waveform, metadata),
    где waveform имеет форму (1, L), а L может отличаться.
    Функция дополняет (паддирует) все аудио до максимальной длины в батче.
    """
    waveforms = [item[0] for item in batch]
    metadatas = [item[1] for item in batch]
    max_length = max(waveform.shape[-1] for waveform in waveforms)
    padded_waveforms = []
    for waveform in waveforms:
        length = waveform.shape[-1]
        if length < max_length:
            pad_size = max_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        padded_waveforms.append(waveform)
    batch_waveforms = torch.stack(padded_waveforms, dim=0)  # (batch, 1, max_length)
    return batch_waveforms, metadatas

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for waveform, _ in tqdm(dataloader, desc="Training", leave=False):
        waveform = waveform.to(device)
        optimizer.zero_grad()
        recon, vq_loss, _ = model(waveform)
        recon_loss = nn.functional.mse_loss(recon, waveform)
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for waveform, _ in tqdm(dataloader, desc="Validation", leave=False):
            waveform = waveform.to(device)
            recon, vq_loss, _ = model(waveform)
            recon_loss = nn.functional.mse_loss(recon, waveform)
            loss = recon_loss + vq_loss
            running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    setup_logging("training.log")
    
    # Загружаем датасет с использованием CommonVoiceRussianDataset
    dataset = CommonVoiceRussianDataset(split="train", min_length=16000)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Используем custom collate_fn для обработки переменной длины аудио
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    
    model = VQVAE(in_channels=1, latent_dim=64, num_embeddings=512, commitment_cost=0.25)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    best_epoch = -1
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        logging.info("Epoch %d: Train Loss = %.6f, Val Loss = %.6f", epoch+1, train_loss, val_loss)
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_filename)
        logging.info("Checkpoint saved: %s", checkpoint_filename)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch+1
            torch.save(model.state_dict(), "best_vqvae.pth")
            logging.info("New best model at epoch %d with Val Loss: %.6f", epoch+1, val_loss)
    logging.info("Training completed. Best model at epoch %d with Val Loss: %.6f", best_epoch, best_val_loss)

if __name__ == "__main__":
    main()
