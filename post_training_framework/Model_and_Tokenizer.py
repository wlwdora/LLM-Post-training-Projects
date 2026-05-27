import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_model_and_tokenizer(model_name="Qwen/Qwen2.5-2B-Instruct"):
    # 1. 量化配置 (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # 计算精度
        bnb_4bit_use_double_quant=True,
    )

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",           
        trust_remote_code=True,       
        attn_implementation="flash_attention_2" 
    )

    # 4. 准备模型用于k-bit训练 (梯度检查点等)
    model = prepare_model_for_kbit_training(model)

    # 5. 配置 LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. 获取 PEFT 模型
    model = get_peft_model(model, lora_config)

    # 打印可训练参数信息 (面试必问)
    model.print_trainable_parameters()

    return model, tokenizer

# 如果单独运行此脚本进行测试
if __name__ == "__main__":
    model, tokenizer = get_model_and_tokenizer()
    print("模型和分词器加载成功！")