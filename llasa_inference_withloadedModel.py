from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
import os
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def ids_to_speech_tokens(speech_ids):
    
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

def extract_speech_ids(speech_tokens_str):

    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

from xcodec2.modeling_xcodec2 import XCodec2Model
 
model_path = "HKUST-Audio/xcodec2"  
local_xcodec2_save_dir = './pretrained_model/xcodec2_local'
 
print ('loading xcodec2 model')
if not os.path.exists(local_xcodec2_save_dir):

    Codec_model = XCodec2Model.from_pretrained(model_path)
    Codec_model.save_pretrained(local_xcodec2_save_dir)
else:
    Codec_model = XCodec2Model.from_pretrained(local_xcodec2_save_dir)
Codec_model.eval().to(device)
# only 16khz speech support!



llasa_3b = './pretrained_model/checkpoint-10205'
# llasa_3b ='HKUSTAudio/Llasa-1B'
print ('loading llasa model')
tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
model = AutoModelForCausalLM.from_pretrained(llasa_3b)
model.eval() 
model.to(device)


 

# prompt_wav, sr = sf.read("longer_prompt.wav") #
# prompt_text = 'I have quite the stomachache. Sungo, can i take the day off?'

prompt_wav, sr = sf.read("doctor_prompt.wav") #
prompt_text = "A long time ago, i made a major decision world."
 

prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)  


while True:
    input_text1 = input("Please enter the  text  to read (type 'quit' to exit): ")
    if input_text1.lower() == 'quit':
        break

    # Get the second input
    input_text2 = input("Please enter the speed from ['fast' and 'slow'] (type 'quit' to exit): ")
    if input_text2.lower() == 'quit':
        break

    elif input_text2.lower() not in ['fast', 'slow']:
        print("Invalid speed. Please enter 'fast' or 'slow'.")
        continue
    else:
        if input_text2.lower() == 'fast':
            speed_token = '<|FAST|>'
        else:
            speed_token = '<|SLOW|>'
    input_text3 = input("Please enter save file name(type 'quit' to exit): ")
    if input_text3.lower() == 'quit':
        break
    else:
        save_file = input_text3


    target_text = input_text1

 

    input_text = prompt_text + speed_token  + target_text

    

    #TTS start!
    print ("Prompt Text:", input_text)
    with torch.no_grad():
        # Encode the prompt wav
        vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
        print("Prompt Vq Code Shape:", vq_code_prompt.shape )   

        vq_code_prompt = vq_code_prompt[0,0,:]
        # Convert int 12345 to token <|s_12345|>
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
        print (speech_ids_prefix[:10])

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text and the speech prefix
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat, 
            tokenize=True, 
            return_tensors='pt', 
            continue_final_message=True
        )
        input_ids = input_ids.to(device)
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate the speech autoregressively
        outputs = model.generate(
            input_ids,
            max_length=2048,  # We trained our model with a max length of 2048
            eos_token_id= speech_end_id ,
            do_sample=True,
            top_p=1,           
            temperature=0.9,  # Lower temperature for more stable results
            pad_token_id=tokenizer.eos_token_id,
        )
        # Extract the speech tokens
        # generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
        generated_ids = outputs[0][input_ids.shape[1]:-1]

        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   

        # Convert  token <|s_23456|> to int 23456 
        speech_tokens = extract_speech_ids(speech_tokens)

        speech_tokens = torch.tensor(speech_tokens).to(device).unsqueeze(0).unsqueeze(0)

        # Decode the speech tokens to speech waveform
        gen_wav = Codec_model.decode_code(speech_tokens) 

 

    sf.write(f"gen_{input_text3}.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)
    print (f"Generated speech saved as gen_{input_text3}.wav")
    print ("---------------------------------------------")
