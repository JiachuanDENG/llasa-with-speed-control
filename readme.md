# LLASA With Speed Control

#### Update (2025-03-15): Add Llasa Speed control finetune and dataprocessing code.

#### On-going experiments: Emotion Contorl & Accent Control

## Introduction
This project is a fine-tuned version of the original LLASA project. The original project was developed by the [LLaSA_training](https://github.com/zhenye234/LLaSA_training) 

The original LLASA model can already generate very high quality and natural sounding speech. However, the original model does not have the ability to control the speed of the generated speech. This project aims to show LLASA can also be extended to include more controllable features, included but not limited to speed control.

## Dataset
We used the open-sourced Genshin Impact dataset from [here](https://pan.ai-hobbyist.com/Genshin%20Datasets/%E8%8B%B1%E8%AF%AD%20-%20English) to fine-tune the original LLASA model.

## Basic Idea
The basic idea is simple, we roughly divide the Genshin Impact dataset into 3 categories: slow, medium, and fast based on syllable count per second. And add special tokens ['<|SLOW|>', '<|MEDIUM|>', '<|FAST|>'] to control the generated speech speed.  

See details in `data_preprocessing/get_memmap_speedV2.py`

## Try the model
Note that huggingface tokens is necessary to download xcodec2 pre-trained model. 

The fine-tuned LLASA model can be downloaded [here](https://drive.google.com/drive/folders/13BTLFwp8ua96-y_eehHJ1eoUFXP1Tfoq?usp=sharing).
Place the downloaded model under `pretrained_model` folder and run the following command:
```bash
python llasa_inference_withloadedModel.py
```
Type in the text you want to generate and the speed you want to control. The generated speech will be saved as `./example/gen_outs/gen_[filename]_[speed].wav` in the same directory. 
![Image](https://github.com/user-attachments/assets/dcaa8230-c5c8-4350-88e5-58daacefe299)




