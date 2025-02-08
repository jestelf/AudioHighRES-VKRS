# compress_decompress.py
import argparse
import json
import numpy as np
import torch
import torchaudio
import soundfile as sf
from model import VQVAE

def preprocess_audio(audio_path, min_length=16000, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] < min_length:
        pad_size = min_length - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    waveform = waveform.unsqueeze(0)  # (1, L)
    return waveform

def compress_audio_func(model, audio_path, device="cpu"):
    waveform = preprocess_audio(audio_path)
    waveform = waveform.to(device)
    with torch.no_grad():
        _, vq_loss, quantized = model(waveform)
    return quantized.cpu().numpy()

def decompress_audio_func(model, quantized, device="cpu"):
    quantized_tensor = torch.tensor(quantized, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = model.decoder(quantized_tensor)
    reconstructed = reconstructed.cpu().squeeze(0)
    return reconstructed

def main():
    parser = argparse.ArgumentParser(description="Compress or decompress audio using a trained VQVAE.")
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True,
                        help="Mode: compress (compress audio) or decompress (restore audio)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file path (audio file for compress, npz file for decompress)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path (npz for compress, audio file for decompress)")
    parser.add_argument("--metadata", type=str, default="", help="JSON string with metadata (for compress)")
    parser.add_argument("--checkpoint", type=str, default="best_vqvae.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(in_channels=1, latent_dim=64, num_embeddings=512, commitment_cost=0.25)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    if args.mode == "compress":
        quantized = compress_audio_func(model, args.input, device=device)
        metadata = {}
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except Exception as e:
                metadata = {"info": args.metadata}
        np.savez(args.output, quantized=quantized, metadata=json.dumps(metadata))
        print("Compression completed. Saved to:", args.output)
    elif args.mode == "decompress":
        data = np.load(args.input, allow_pickle=True)
        quantized = data["quantized"]
        metadata = json.loads(data["metadata"].item())
        reconstructed = decompress_audio_func(model, quantized, device=device)
        sf.write(args.output, reconstructed.T, 16000)
        print("Decompression completed. Reconstructed audio saved to:", args.output)
        print("Metadata:", metadata)
    
if __name__ == "__main__":
    main()
