
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
import os 
from tqdm import tqdm 
import resampy
import numpy as np
 

from xcodec2.modeling_xcodec2 import XCodec2Model
 
model_path = "HKUST-Audio/xcodec2"  
local_xcodec2_save_dir = '/share5/users/jiachuan/code/Llasa/xcodec2_local'

if not os.path.exists(local_xcodec2_save_dir):
    Codec_model = XCodec2Model.from_pretrained(model_path)
else:
    Codec_model = XCodec2Model.from_pretrained(local_xcodec2_save_dir)
Codec_model.eval().cuda()   


input_original = '/share5/users/jiachuan/data/llasa_ft_data/Genshin/wav/'
output_code = '/share5/users/jiachuan/data/llasa_ft_data/Genshin/vq_codes/'
if not os.path.exists(output_code):
    os.makedirs(output_code)

for file in tqdm(os.listdir(input_original)[:]):
    if not file.endswith(".wav"): continue

    audio_path = os.path.join(input_original, file)
    output_path = os.path.join(output_code, file+ ".npy")

    prompt_wav, sr = sf.read(audio_path)   # you can find wav in Files
    if sr != 16000:
        # Resample the audio to 16kHz
        prompt_wav = resampy.resample(prompt_wav, sr, 16000)
        sr = 16000
    #prompt_wav, sr = sf.read("Anna.wav") # English prompt
    prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)  
    with torch.no_grad():
        # Encode the prompt wav
        vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
        # print("Prompt Vq Code Shape:", vq_code_prompt.shape )   

        vq_code_prompt = vq_code_prompt[0,0,:]
        np.save(output_path, vq_code_prompt.cpu().numpy())
     