# PPOtrain.py
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset

# 1. 加载模型
# PPO 需要带有 Value Head 的模型来充当 Critic
model = AutoModelForCausalLMWithValueHead.from_pretrained("./final_sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("./final_sft_model") # 冻结的参考模型
tokenizer = AutoTokenizer.from_pretrained("./final_sft_model")

# 2. 加载奖励模型（用于在训练中实时打分）
from transformers import AutoModelForSequenceClassification
reward_model = AutoModelForSequenceClassification.from_pretrained("./final_rm_model")

# 3. PPO 配置
ppo_config = PPOConfig(
    #基础信息
    model_name="qwen-7b-sft",           # 模型名称（用于wandb记录或标识）
    learning_rate=1.41e-5,              # Actor模型的学习率
    batch_size=8,                       # 每次Step处理的总样本数（全局批次）
    mini_batch_size=2,                  # 每个Mini-batch的大小（用于多次更新）
    optim="paged_adamw_32bit",              # 显式指定优化器
    ppo_epochs=4,                       # 每批数据进行几轮更新（通常3-4轮）
    
    #核心缺失项补全
    output_dir="./output_qwen_ppo",     # 【关键】模型保存和日志输出的路径
    gradient_checkpointing=True,        # 【关键】开启梯度检查点，显存不够时必须开！
    
    #关键超参数优化
    init_kl_coef=0.2,                   # KL散度系数（惩罚项），防止模型跑偏
    target_kl=6.0,                      # 目标KL阈值，用于自适应调整系数
    gamma=1.0,                          # 折扣因子（RL中通常设为1，因为是单步决策）
    lam=0.95,                           # GAE计算优势时的平滑系数
    clip_range=0.2,                     # PPO的核心裁剪范围，限制更新幅度
    clip_range_value=0.2,               # Value Head的裁剪范围
    vf_coef=0.1,                        # Value Loss的权重系数
    max_grad_norm=1.0,                  # 梯度裁剪，防止梯度爆炸
    
    #日志与保存 
    log_with="wandb",                   # 也可以用 "tensorboard"
    project_name="qwen-rlhf",           # 项目名称
    steps=20000,                        # 总训练步数
    save_freq=100,                      # 每多少步保存一次检查点
    seed=42,                            # 随机种子，保证复现性
)

# 5. 准备纯 Prompt 数据集（PPO 只需要 prompt，回答由模型自己生成）
prompt_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

ppo_trainer = PPOTrainer(
    config=ppo_config, 
    model=model,          
    ref_model=ref_model,  
    tokenizer=tokenizer, 
    dataset=prompt_dataset, 
    reward_model=reward_model  
)

ppo_trainer.train() 