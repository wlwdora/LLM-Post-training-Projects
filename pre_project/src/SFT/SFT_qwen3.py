import torch
from swift import sft_main, SftArguments

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 使用设备: {device}")

# ========== Sft配置==========
args = SftArguments(
    model='D:/Qwen/Qwen3-VL-2B-Instruct',
    model_type='qwen3_vl',
    dataset='./SFT_data_fixed.jsonl',
    output_dir='./output/qwen3_vl_gesture',
    
    # 训练类型
    tuner_type='lora',
    
    # LoRA 参数
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    
    # === 核心性能 ===
    bf16=True,      # 强制开启 BF16
    fp16=False,     # 强制关闭 FP16，防止冲突
    per_device_train_batch_size=4,      
    gradient_accumulation_steps=2,      
    dataloader_num_workers=0,           
    
    eval_steps=100,                     
    save_steps=100,                    
    logging_steps=20,                   
    
    # 其他训练参数
    learning_rate=2e-5,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    warmup_ratio=0.15,
    lr_scheduler_type='cosine',
    gradient_checkpointing=True,
    
    # 数据集划分
    split_dataset_ratio=0.1,
    
    # 保存最优模型
    load_best_model_at_end=True,
    
    # 日志
    report_to='tensorboard',
)

# ========== 3. 开始训练 ==========
if __name__ == '__main__':
    print("🏋️ 开始训练...")
    sft_main(args)
    print("✅ 训练完成！")