"""
SFT → DPO 数据集构建脚本
功能：
  1. 从SFT JSONL中按类别（石头/剪刀/布/无法识别）采样
  2. 生成chosen: <answer>标签</answer>（简洁正确）
  3. 生成三种rejected：
     - verbose:  XML标签内包含冗余啰嗦内容
     - wrong:    XML标签内包含错误类别答案
     - broken:   XML标签格式残缺（缺头/缺尾/标签名错误等）
  4. 输出标准DPO格式（prompt / chosen / rejected）
"""

import json
import random
import argparse
from collections import defaultdict

random.seed(42)

# ==================== 配置区 ====================
# 原始SFT数据路径
INPUT_PATH = "SFT_data_fixed.jsonl"

# 输出DPO数据路径
OUTPUT_PATH = "DPO_dataset.jsonl"

# 每类采样数量
SAMPLE_PER_CLASS = 1000

# 四种类别及其在SFT中可能出现的原始回答变体
LABEL_CATEGORIES = {
    "石头": ["石头", "答案是：石头", "根据图片，石头", "图中手势是：石头"],
    "剪刀": ["剪刀", "答案是：剪刀", "根据图片，剪刀", "图中手势是：剪刀"],
    "布":   ["布",   "答案是：布",   "根据图片，布",   "图中手势是：布"],
    "无法识别": ["无法识别", "根据图片，无法识别", "图中手势是：无法识别", "答案是：无法识别"],
}

CATEGORIES = list(LABEL_CATEGORIES.keys())

# 冗余回答模板（会被包裹在<answer>...</answer>中）
VERBOSE_TEMPLATES = [
    "我仔细观察了图片，经过认真分析和判断，这个手势应该是{label}",
    "根据图片中的手势特征进行详细分析后，我认为答案是{label}",
    "我查看了这张图片，经过思考，判断图中比出的手势是{label}",
    "从图片内容来看，经过我的识别和分析，这个手势代表{label}",
    "通过对图片中手势的仔细观察和辨认，我确定这个手势是{label}",
]

# 残缺标签格式（内容正确，标签残缺）
BROKEN_TAG_TEMPLATES = [
    "<answer>{label}",              # 缺结束标签
    "{label}</answer>",              # 缺开始标签
    "<ans>{label}</answer>",         # 开始标签名错误
    "<answer>{label}</ans>",         # 结束标签名错误
    "< answer>{label}</answer>",    # 开始标签有空格
    "<answer>{label}< /answer>",    # 结束标签有空格
    "{label}",                      # 无标签
]

REJECTED_TYPES = ["verbose", "wrong", "broken"]
# ================================================


def generate_dpo_record(record, category, rejected_type):
    """根据记录、类别和rejected类型生成单条DPO数据"""
    prompt = record["messages"][0]  # user message（保留image + text）

    # chosen: 简洁正确，严格XML标签（修改为列表格式）
    chosen = {"role": "assistant", "content": [{"type": "text", "text": f"<answer>{category}</answer>"}]}

    # rejected: 三种缺陷类型
    if rejected_type == "verbose":
        text = random.choice(VERBOSE_TEMPLATES).format(label=category)
        rejected_content = f"<answer>{text}</answer>"

    elif rejected_type == "wrong":
        wrong_label = random.choice([c for c in CATEGORIES if c != category])
        rejected_content = f"<answer>{wrong_label}</answer>"

    elif rejected_type == "broken":
        rejected_content = random.choice(BROKEN_TAG_TEMPLATES).format(label=category)

    else:
        raise ValueError(f"Unknown rejected_type: {rejected_type}")

    # rejected: 修改为列表格式
    rejected = {"role": "assistant", "content": [{"type": "text", "text": rejected_content}]}

    return {
        "prompt": [prompt],
        "chosen": [chosen],
        "rejected": [rejected]
    }


def main():
    parser = argparse.ArgumentParser(description="Build DPO dataset from SFT data")
    parser.add_argument("--input", "-i", default=INPUT_PATH, help="输入SFT JSONL路径")
    parser.add_argument("--output", "-o", default=OUTPUT_PATH, help="输出DPO JSONL路径")
    parser.add_argument("--num", "-n", type=int, default=SAMPLE_PER_CLASS, help="每类采样数量")
    args = parser.parse_args()

    # 1. 读取SFT数据
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[INFO] 读取SFT记录: {len(records)} 条")

    # 2. 建立原始标签 -> 标准类别的映射
    label_to_category = {}
    for cat, labels in LABEL_CATEGORIES.items():
        for lbl in labels:
            label_to_category[lbl] = cat

    # 3. 按类别分组
    category_records = defaultdict(list)
    for r in records:
        orig_label = r["messages"][1]["content"]
        cat = label_to_category.get(orig_label)
        if cat:
            category_records[cat].append(r)

    print("[INFO] 各类别分布:")
    for cat, recs in category_records.items():
        print(f"       {cat}: {len(recs)} 条")

    # 4. 每类采样并生成DPO
    dpo_records = []
    for cat in CATEGORIES:
        recs = category_records[cat]
        n = min(args.num, len(recs))
        sampled = random.sample(recs, n)
        print(f"[INFO] {cat} 采样 {n} 条")

        for i, rec in enumerate(sampled):
            rej_type = REJECTED_TYPES[i % 3]
            dpo_records.append(generate_dpo_record(rec, cat, rej_type))

    # 5. 打乱并输出
    random.shuffle(dpo_records)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] DPO数据集已生成: {args.output}")
    print(f"[INFO] 总计: {len(dpo_records)} 条")
    print("[INFO] rejected类型占比: verbose≈33%, wrong≈33%, broken≈34%")


if __name__ == "__main__":
    main()