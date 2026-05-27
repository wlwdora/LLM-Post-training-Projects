# DPOtrain.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from DPO_Datasets_gen import get_dpo_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# 1. 加载 SFT 训练好的模型作为起点
model_name = "./final_sft_model"  # 使用你上一步 SFT 产出的模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 准备 DPO 数据集
dataset = get_dpo_dataset(tokenizer)

# 3. 配置 LoRA（DPO 同样支持 PEFT 高效微调）
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# 4. DPO 专属训练配置
dpo_config = DPOConfig(
    #核心缺失项补全
    output_dir="./dpo_output",        # 保存路径
    run_name="qwen-7b-dpo-v1",        # 实验名称（用于wandb或区分不同实验）
    seed=42,                          # 保证复现性
    gradient_checkpointing=True,      # 开启梯度检查点
    gradient_checkpointing_kwargs={"use_reentrant": False}, # 避免部分模型报错
    
    #训练超参数
    num_train_epochs=1,               # DPO通常跑1or2个Epoch就够了，多了容易过拟合
    per_device_train_batch_size=2,    # 单卡批次大小
    gradient_accumulation_steps=8,    # 梯度累积步数 
    learning_rate=5e-6,               # DPO学习率通常比SFT低
    beta=0.1,                         # KL控制系数，控制偏离参考模型的程度
    warmup_ratio=0.1,                 # 预热比例
    
    #硬件与精度优化
    bf16=True,                        # 使用bfloat16混合精度，比fp16更稳定
    optim="paged_adamw_32bit",        # 显式指定优化器
    lr_scheduler_type="cosine",       # 学习率调度策略
    max_length=1024,                  # 序列最大长度

    #日志与保存 
    logging_steps=10,                 # 每10步打印一次日志
    save_steps=500,                   # 每500步保存一次检查点
    save_total_limit=2,               # 只保留最新的2个检查点，防止硬盘爆满
    eval_strategy="no",               # DPO很难做验证集评估，通常设为no，或者steps
    
    remove_unused_columns=False,    
    report_to="none",               
)

# 5. 初始化 DPOTrainer
# ref_model=None 时，TRL 会自动基于当前 model 创建一个冻结的参考模型
trainer = DPOTrainer(
    model=model,
    ref_model=None,             
    args=dpo_config,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

# 6. 开始训练
trainer.train()

# 7. 保存最终模型
trainer.save_model("./dpo_final_model")