# test_model.py
import torch
import torchaudio
import soundfile as sf
from model import VQVAE
from dataset import CommonVoiceRussianDataset

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(in_channels=1, latent_dim=64, num_embeddings=512, commitment_cost=0.25)
    checkpoint = "best_vqvae.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    dataset = CommonVoiceRussianDataset(split="test", segment_length=16000)
    waveform, metadata = dataset[0]
    print("Test sample metadata:", metadata)
    
    waveform = waveform.to(device)
    with torch.no_grad():
        recon, vq_loss, quantized = model(waveform.unsqueeze(0))
    recon = recon.cpu().squeeze(0)
    sf.write("reconstructed_test.wav", recon.T, 16000)
    print("Reconstructed test audio saved to reconstructed_test.wav")

if __name__ == '__main__':
    test_model()
