# app.py
import os
import torch
import torchaudio
import gradio as gr
from configs import Config
from models import VoiceCloningTTS
from text_preprocessing import normalize_text

model_ref=[None]
map_ref=[None]
loaded=[False]

def load_ckpt(ckpt):
    st=torch.load(ckpt, map_location=Config.DEVICE)
    m=VoiceCloningTTS(len(st["char_map"])+1,len(st["spk_map"])+1,len(st["emo_map"])+1)
    m.load_state_dict(st["model"])
    m.eval().to(Config.DEVICE)
    return m, st["char_map"]

def t2t(txt, cm):
    arr=[]
    for c in txt:
        arr.append(cm[c] if c in cm else 0)
    if not arr:
        arr=[0]
    return torch.LongTensor(arr).unsqueeze(0)

def inference_fn(text,audio_ref):
    if not loaded[0]:
        ck=os.path.join(Config.CHECKPOINT_DIR,"model_ep0.pth")
        mod,mp=load_ckpt(ck)
        model_ref[0]=mod
        map_ref[0]=mp
        loaded[0]=True
    m=model_ref[0]
    c=map_ref[0]
    nt=normalize_text(text)
    ts=t2t(nt,c).to(Config.DEVICE)
    ar=torch.from_numpy(audio_ref[0])
    sr=audio_ref[1]
    if ar.dim()==2:
        ar=ar.mean(dim=0)
    if sr!=Config.SAMPLE_RATE:
        ar=torchaudio.functional.resample(ar,sr,Config.SAMPLE_RATE)
    ar=ar.unsqueeze(0)
    me=torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_mels=Config.N_MEL_CHANNELS,
        n_fft=Config.N_FFT,
        win_length=Config.WIN_LENGTH,
        hop_length=Config.HOP_LENGTH
    )
    rm=me(ar).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        mo,wo,_,_=m(rm,ts)
    o=wo.squeeze(0).cpu().unsqueeze(0)
    return (Config.SAMPLE_RATE,o.numpy())

def launch_ui():
    with gr.Blocks() as demo:
        t=gr.Textbox(label="Текст")
        a=gr.Audio(source="microphone",type="numpy",label="Референс (аудио)")
        b=gr.Button("Сгенерировать")
        out=gr.Audio(label="Результат")
        b.click(fn=inference_fn,inputs=[t,a],outputs=out)
    demo.launch()

if __name__=="__main__":
    os.makedirs(Config.CHECKPOINT_DIR,exist_ok=True)
    launch_ui()
