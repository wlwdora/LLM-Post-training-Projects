
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift import RLHFArguments, rlhf_main
from swift.tuners import Swift

# ==================== 配置区 ====================
MODEL_PATH = "D:/Qwen/output/merged_sft_qwen_model_swift"  # 基础模型路径
DATASET_PATH = "./dpo_dataset_swift_final.jsonl"            # DPO数据集路径
OUTPUT_DIR = "./output/dpo_lora"              # DPO输出目录

# DPO超参数
BETA = 0.1                    # KL惩罚系数，越大越保守
LEARNING_RATE = 5e-6          # DPO学习率（比SFT低一    个数量级）
NUM_EPOCHS = 2                # DPO训练轮数
BATCH_SIZE = 1                # 单卡batch size（多模态建议1）
GRAD_ACCUM_STEPS = 4          # 梯度累积步数
MAX_LENGTH = 512             # 最大序列长度
LORA_RANK = 4                 # LoRA秩
LORA_ALPHA = 8               # LoRA缩放系数
# ================================================

def main():
    # 构建训练参数
    # ms-swift 3.x 的 rlhf 入口使用 TrainArguments + rlhf_type="dpo"
    args = RLHFArguments(
        # 模型配置
        model=MODEL_PATH,
        model_type="qwen3_vl",

        # 训练类型
        tuner_type="lora",           # lora / full / qlora
        rlhf_type="dpo",             # dpo / orpo / simpo / kto / cpo

        # 数据集
        dataset=[DATASET_PATH],

        # DPO特定参数
        beta=BETA,

        # LoRA配置
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["all-linear"],  # 或指定具体模块

        # 训练参数
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,

        # 优化器与调度
        optim="adamw_torch",
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        weight_decay=0.01,

        # 显存优化
        gradient_checkpointing=True,
        torch_dtype="float16",      # bfloat16 / float16

        # 工程参数
        dataloader_num_workers=0,      # 多模态建议0，避免多进程卡死
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        remove_unused_columns=False,   # 多模态必须False，保留image数据

    )

    # 启动DPO训练
    print("=" * 60)
    print("开始 DPO 训练")
    print(f"模型: {MODEL_PATH}")
    print(f"数据集: {DATASET_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"Beta: {BETA}")
    print(f"学习率: {LEARNING_RATE}")
    print("=" * 60)

    result = rlhf_main(args)
    print(f"训练完成，模型保存在: {OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    main()