 
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing
import random
 
def init_worker(transcriptions_, code_root_, tokenizer_, max_seq_len_, base_num_,speaker_speed_map_):
    global transcriptions, code_root, tokenizer, max_seq_len, base_num,speech_rate
    transcriptions = transcriptions_
    code_root = code_root_
    tokenizer = tokenizer_
    max_seq_len = max_seq_len_
    base_num = base_num_
    speaker_speed_map = speaker_speed_map_

def process_audio_id(audio_id):
 
    transcript = transcriptions.get(audio_id)
    spk_name = audio_id.split('_')[0]
    slow_audio_ids = speaker_speed_map[spk_name]['slow']
    fast_audio_ids = speaker_speed_map[spk_name]['fast']

    fast_or_slow = random.choice(['fast','slow'])
    if fast_or_slow == 'fast':
        speed_id = speed_token_id['<|FAST|>']
        speed_audio_id = random.choice(fast_audio_ids)
        speed_transcript = transcriptions.get(speed_audio_id)
    else:
        speed_id = speed_token_id['<|SLOW|>']
        speed_audio_id = random.choice(slow_audio_ids)
        speed_transcript = transcriptions.get(speed_audio_id)

    if transcript is None or speed_transcript is None:
        return None   

 
    code_file = os.path.join(code_root, audio_id + '.wav.npy')
    if not os.path.exists(code_file):
        return None
    try:
        codes = np.load(code_file, allow_pickle=True)
    except Exception as e:
        print(f"load {code_file}: {e}")
        return None
    if codes.ndim != 1:
        print(f"The shape of {code_file} is wrong: {codes.ndim}")
        return None
    codes = base_num + torch.tensor(codes, dtype=torch.long)


    code_speed_file = os.path.join(code_root, speed_audio_id + '.wav.npy')
    if not os.path.exists(code_speed_file):
        return None
    try :
        codes_speed = np.load(code_speed_file, allow_pickle=True)
    except Exception as e:
        print(f"load {code_speed_file}: {e}")
        return None
    if codes_speed.ndim != 1:
        print(f"The shape of {code_speed_file} is wrong: {codes_speed.ndim}")
        return None
    codes_speed = base_num + torch.tensor(codes_speed, dtype=torch.long)
 
 


    text_with_special_part1 = f"<|TEXT_UNDERSTANDING_START|>{transcript}"
    encoded_text_part1 = tokenizer.encode_plus(
        text_with_special_part1,
        add_special_tokens=False,
        return_tensors='np'
    )
    text_input_ids_part1 = encoded_text_part1['input_ids'].squeeze(0)  # (text_len,)

    text_with_special_part2 = f"{speed_transcript}<|TEXT_UNDERSTANDING_END|>"
    encoded_text_part2 = tokenizer.encode_plus(
        text_with_special_part2,
        add_special_tokens=False,
        return_tensors='np'
    )
    text_input_ids_part2 = encoded_text_part2['input_ids'].squeeze(0)  # (text_len,)



   
    text_input_ids = np.array( text_input_ids_part1.tolist() + [speed_id] +  text_input_ids_part2.tolist() , dtype=np.int32)
 
    speech_gen_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    speech_gen_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    code_input_ids = np.array(  
        [speech_gen_start_id] +
        codes.tolist() + codes_speed.tolist() +
        [speech_gen_end_id],
        dtype=np.int32
    )

 
    total_input_ids = np.concatenate([text_input_ids, code_input_ids])

 
    if len(total_input_ids) > max_seq_len:
        # total_input_ids = total_input_ids[:max_seq_len]
        return None
    else:
        padding_length = max_seq_len - len(total_input_ids)
        total_input_ids = np.pad(
            total_input_ids,
            (0, padding_length),
            'constant',
            constant_values=tokenizer.pad_token_id
        )

    return total_input_ids.astype(np.int32)

def process_data(transcriptions, code_root, output_dir_tts, num_processes=4,speaker_speed_map=None):
    max_seq_len = 2048
 
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-1B-Instruct',
        model_max_length=2048,
        padding_side="right",
    )
 
    tokenizer.pad_token = tokenizer.eos_token

 
 
    special_tokens = [
        '<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>',
        '<|TEXT_UNDERSTANDING_START|>', '<|TEXT_UNDERSTANDING_END|>',
        '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
        '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>'
    ]
    tokenizer.add_tokens(special_tokens)
    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
 
    base_num = len(tokenizer)
    # print (base_num)
    # exit()


    

    audio_ids = []
    for spk in speaker_speed_map:
        for audio_id in speaker_speed_map[spk]['medium']:
            audio_ids.append(audio_id)
 
    # audio_ids = list(transcriptions.keys())
    np.random.shuffle(audio_ids)
    val_len = int(0.01*len(audio_ids))
    val_audio_ids = audio_ids[-val_len:]
    train_audio_ids = audio_ids[:-val_len]

    num_processes = min(num_processes, multiprocessing.cpu_count())

 
    with multiprocessing.Pool(
        num_processes,
        initializer=init_worker,
        initargs=(transcriptions, code_root, tokenizer, max_seq_len, base_num,speaker_speed_map)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_audio_id, train_audio_ids),
            total=len(train_audio_ids),
            desc="data processing"
        ))
    train_tts_input_ids_list = [res for res in results if res is not None]

 
    init_worker(transcriptions, code_root, tokenizer, max_seq_len, base_num,speaker_speed_map)
    val_tts_input_ids_list = []
    for audio_id in tqdm(val_audio_ids, desc="valid data processing"):
        res = process_audio_id(audio_id)
        if res is not None:
            val_tts_input_ids_list.append(res)
 
    if not (train_tts_input_ids_list or val_tts_input_ids_list):
        print("bug ")
        return

 
    all_ids = train_tts_input_ids_list + val_tts_input_ids_list
    max_total_token_len = max(len(ids) for ids in all_ids)
 
 
    os.makedirs(output_dir_tts, exist_ok=True)

 
    train_tts_input_ids_array = np.array(train_tts_input_ids_list)
    val_tts_input_ids_array = np.array(val_tts_input_ids_list)

    train_tts_memmap_path = os.path.join(output_dir_tts, 'train_input_ids.memmap')
    train_tts_memmap = np.memmap(
        train_tts_memmap_path, dtype='int32', mode='w+', shape=train_tts_input_ids_array.shape
    )
    train_tts_memmap[:] = train_tts_input_ids_array[:]
    del train_tts_memmap   

    val_tts_memmap_path = os.path.join(output_dir_tts, 'val_input_ids.memmap')
    val_tts_memmap = np.memmap(
        val_tts_memmap_path, dtype='int32', mode='w+', shape=val_tts_input_ids_array.shape
    )
    val_tts_memmap[:] = val_tts_input_ids_array[:]
    del val_tts_memmap   

 
    np.save(os.path.join(output_dir_tts, 'train_input_ids_shape.npy'), train_tts_input_ids_array.shape)
    np.save(os.path.join(output_dir_tts, 'val_input_ids_shape.npy'), val_tts_input_ids_array.shape)

    print(f" TTS memmap  saved ! {output_dir_tts}")

if __name__ == "__main__":
 
   
    code_root = '/share5/users/jiachuan/data/llasa_ft_data/Genshin/vq_codes'
    
 
 
    trans_file = '/share5/users/jiachuan/data/llasa_ft_data/Genshin/metadata.csv'
 
    speaker_withDifferent_speed_folder = '/share5/users/jiachuan/data/llasa_ft_data/Genshin/speaker_withDifferent_speed'
 
    output_dir_tts = '/share5/users/jiachuan/data/llasa_ft_data/Genshin/Genshin_speedV2_bin'

    tokenizer = AutoTokenizer.from_pretrained(
       'HKUSTAudio/Llasa-1B',
        model_max_length=2048,
        padding_side="right",
    )
    speed_tokens = [
        '<|FAST|>' ,
            '<|SLOW|>' ,
        '<|MEDIUM|>'
    ]
    tokenizer.add_tokens(speed_tokens)
    speed_token_id = {
          '<|FAST|>': tokenizer.convert_tokens_to_ids('<|FAST|>'),
     
         '<|SLOW|>': tokenizer.convert_tokens_to_ids('<|SLOW|>'),
        '<|MEDIUM|>': tokenizer.convert_tokens_to_ids('<|MEDIUM|>')
    }

 
    transcriptions = {}
    with open(trans_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 2:
                continue
            audio_id = parts[0]
            transcript = parts[1]   
            transcriptions[audio_id] = transcript
    

    speaker_speed_map = {}
    for spk in os.listdir(speaker_withDifferent_speed_folder):
        speaker_speed_map[spk] = {}
        for speed in os.listdir(os.path.join(speaker_withDifferent_speed_folder,spk)):
            speed = speed.split('.txt')[0]
            if speed not in ['fast','slow','medium']:
                continue
            speaker_speed_map[spk][speed] = []
            with open(os.path.join(speaker_withDifferent_speed_folder,spk,speed+'.txt'),'r',encoding='utf-8') as f:
                for line in f:
                    speaker_speed_map[spk][speed].append(line.strip().split('.wav')[0])
            

 
    num_processes = 8

 
    process_data(
        transcriptions,
        code_root,
        output_dir_tts,
        num_processes=num_processes,
        speaker_speed_map = speaker_speed_map
    )
