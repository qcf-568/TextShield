# TextShield
[AAAI2026] TextShield-R1: Reinforced Reasoning for Tampered Text Detection

---

### Enviroment
```
pip install ms-swift transformers qwen-vl-utils[decord]==0.0.8
```
---

### Model Inference 

1. Download the fine-tuned model at [Baidu Cloud](https://pan.baidu.com/s/1sAJ0SkJbCW0MKW-K30tEAg?pwd=text)

2. Run model start code
```
CUDA_VISIBLE_DEVICES=0 swift infer --model [your downloaded textshield model dir] --stream true --max_new_tokens 4096
```
3. Input the prompt and image path
```
<image> Is this image real, entirely generated, or tampered? If it has been tampered, what method was used, and what are the content and bounding box coordinates of the tampered text? Output the thinking process in <think> </think> and \n final answer (number) in <answer> </answer> tags.
```
---

### Model Training

