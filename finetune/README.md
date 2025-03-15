# Fine-tune model for Speed Control
Let's say you aleady has a Genshin dataset stored in `[your path]/Genshin/wav`. Please make sure the wav you are using has been converted in 1 channel 16khz wav format.

## Explore the dataset and Divide the data into different Speed
go to `./data_preprocessing/prepare_speedDataset.ipynb` and run the notebook to prepare the dataset for fine-tuning.  

The original dataset will be sorted into 3 categories: slow, medium, and fast based on syllable count per second. Data with different speed information will be saved in:
    ` [your path]/Genshin/speaker_withDifferent_speed`

## Get xcodec2 format data (vq_code)
run `./data_preprocessing/xcodec2_processwav.py` to convert the data into xcodec2 format. 

The xcodec2 format data will be saved in:
` [your path]//Genshin/vq_codes/`

## Get Memmap format data for LLASA model fine-tuning
run `./data_preprocessing/get_memmap_speedV2.py` to convert the xcodec2 format data into memmap format.

This step will randomly combine the data with different speed from the same speaker into a long data, and add special tokens between the corresponding text.  For example:

speaker1_mediumspeed_wav1.wav: "text1, text2, text3" , [audio_token1, audio_token2, audio_token3]

speaker1_slowspeed_wav1.wav: "text4, text5, text6" , [audio_token4, audio_token5, audio_token6]

-> combined: "text1, text2, text3, <|SLOW|>, text4, text5, text6" , [audio_token1, audio_token2, audio_token3, audio_token4, audio_token5, audio_token6]

This data augmentation helps the model to learn the speed control.

I have also tried directly train model with different speed data with special token as speed tag, for example:

speaker1_slowspeed_wav1.wav: 
"text4, text5, text6" <|SLOW|> [audio_token4, audio_token5, audio_token6]

This way won't work well in the inference. Because the model will totally rely on the speed you used as the audio prompt to generate the rest of the audio.

## Start Fine-tuning
go to `./fine-tune/` and run the following command to start fine-tuning:

```bash
torchrun --nproc_per_node=2 finetune_offline_speed.py
```
 
