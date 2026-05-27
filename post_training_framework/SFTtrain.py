# SFTtrain.py

import torch
from transformers import TrainingArguments, Trainer
# 导入前两个脚本定义的函数
from Model_and_Tokenizer import get_model_and_tokenizer
from SFT_Datasets_gen import get_dataset_and_collator

def main():
    # 1. 获取模型和分词器
    print(">>> 正在加载模型和分词器...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model, tokenizer = get_model_and_tokenizer(model_name)

    # 2. 获取数据集和 Collator
    print(">>> 正在处理数据集...")
    train_dataset, data_collator = get_dataset_and_collator(tokenizer)

    # 3. 配置训练参数
    print(">>> 配置训练参数...")
    training_args = TrainingArguments(
        output_dir="./results_qwen_sft",      
        num_train_epochs=3,                   
        per_device_train_batch_size=2,        
        gradient_accumulation_steps=8,        
        optim="paged_adamw_32bit",            
        logging_steps=10,                     
        learning_rate=2e-4,                   
        fp16=True,                            # 如果显卡不支持BF16，使用FP16
        bf16=False,                           
        gradient_checkpointing=True,          # 开启梯度检查点，省显存
        save_steps=100,                       # 保存步数
        save_total_limit=2,                   
        report_to="none",                    
        deepspeed="ds_config.json", 
    )

    # 4. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 5. 开始训练
    print(">>> 开始训练...")
    trainer.train()

    # 6. 保存最终模型
    print(">>> 保存模型...")
    model.save_pretrained("./final_sft_model")
    tokenizer.save_pretrained("./final_sft_model")

if __name__ == "__main__":
    main()