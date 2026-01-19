# TextShield
[AAAI2026] TextShield-R1: Reinforced Reasoning for Tampered Text Detection

---

### Enviroment
```
pip install ms-swift transformers qwen-vl-utils[decord]==0.0.8

cp orm.py [your original ms-swift orm file, e.g. /usr/local/lib/python3.10/dist-packages/swift/plugin/orm.py]
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
4. OCR rectification


OCR rectification is integrated directly in the IoU evaluation scriptx.
```
unzip ocr_info.zip
python eval_iou_with_ocr_rectification.py --input [your inference output json file]
```

5. Model inference with json-dataset-in and json-dataset-out

Please refer to the (ms-swift inference document](https://swift.readthedocs.io/zh-cn/latest/Instruction/Inference-and-deployment.html#id2)

```
CUDA_VISIBLE_DEVICES=0 swift infer --model [your downloaded textshield model dir] --val_dataset [your dataset json file] --max_new_tokens 4096
```


---

### Model Training

1. Pre-training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 swift sft --model Qwen2.5-VL-7B-Instruct --dataset [your pre-training dataset json file] --split_dataset_ratio 0.0001 --train_type lora --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --learning_rate 2e-4 --lora_rank 32 --lora_alpha 64 --target_modules all-linear --freeze_vit False --freeze_aligner False --gradient_accumulation_steps 1 --eval_steps 5000 --save_steps 5000 --save_total_limit 3 --logging_steps 50 --max_length 4096 --output_dir [your output dir] --warmup_ratio 0.03 --dataloader_num_workers 16 --weight_decay 0 --deepspeed zero2

CUDA_VISIBLE_DEVICES=0 swift export --adapters [your pre-training stage output dir] --merge_lora true
```
2. Cold start
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 swift sft --model [your pre-trained model] --dataset "aaai_train.jsonl#12983" --split_dataset_ratio 0.0001 --train_type lora --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --learning_rate 2e-4 --lora_rank 32 --lora_alpha 64 --target_modules all-linear --freeze_vit False --freeze_aligner False --gradient_accumulation_steps 1 --eval_steps 5000 --save_steps 5000 --save_total_limit 3 --logging_steps 50 --max_length 4096 --output_dir [your output dir] --warmup_ratio 0.03 --dataloader_num_workers 16 --weight_decay 0 --deepspeed zero2

CUDA_VISIBLE_DEVICES=0 swift export --adapters [your cold-start stage output dir] --merge_lora true
```
3. GRPO
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 swift rlhf --rlhf_type grpo \
    --model [your cold-start model] \
    --reward_funcs realfake method ocr iou format\
    --reward_weights 1.0 0.5 1.0 1.0 0.1 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.8 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset 'aaai_train.jsonl#51690' \
    --split_dataset_ratio 0.001 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --freeze_vit False \
    --freeze_aligner False \
    --lora_rank 64 \
    --lora_alpha 128 \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --output_dir [your GRPO output dir] \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'You are a helpful assistant.' \
    --deepspeed zero2 \
    --log_completions true \
    --beta 0.001 \
    --num_iterations 1

CUDA_VISIBLE_DEVICES=0 swift export --adapters [your GRPO stage output dir] --merge_lora true
```
The final output dir is the TextShield-R1 model and can be used as the input model dir for the above inference stage.

---

### TFR Benchmark

The benchmark is uploading and releasing
