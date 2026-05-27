# SFT_Datasets_gen.py

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

def get_dataset_and_collator(tokenizer, dataset_path="tatsu-lab/alpaca", max_length=512):
    # 1. 加载数据集 (这里使用alpaca作为示例)
    dataset = load_dataset(dataset_path, split="train[:1000]") # 仅取1000条做演示

    # 2. 格式化 Prompt (Alpaca格式)
    def format_prompt(examples):
        # 这里模仿 Alpaca 的指令格式
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        prompts = []
        for instr, inp, out in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n{out}"
            else:
                prompt = f"### Instruction:\n{instr}\n### Response:\n{out}"
            prompts.append(prompt)
            
        return {"text": prompts}

    # 使用 map 进行批量处理
    dataset = dataset.map(format_prompt, batched=True, remove_columns=dataset.column_names)

    # 3. 分词
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False, # DataCollator 会处理动态 padding
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. 定义 DataCollator (动态填充，提高训练效率)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # 因果语言模型 (CLM) 设为 False
    )

    return tokenized_dataset, data_collator
