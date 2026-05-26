import json
import random
import uuid
import shutil
from pathlib import Path
from collections import Counter

# ========== 配置 ==========
INPUT_JSONL = "./SFT_data_abs.jsonl"
OUTPUT_JSONL = "./SFT_data_fixed.jsonl"
INPUT_IMG_DIR = Path("./SFT_data")
OUTPUT_IMG_DIR = Path("./SFT_data_random")
OUTPUT_IMG_DIR.mkdir(exist_ok=True)

# 多样化的 query 模板
QUERY_TEMPLATES = [
    "这是什么手势？",
    "图中比出了什么？",
    "石头、剪刀还是布？",
    "请识别这个手势",
    "图片里是什么？",
    "这个人出了什么？",
    "判断一下手势",
    "石头剪刀布？",
    "这是什么？",
    "",  # 空 query，强制只看图
    "看图回答",
    "手势识别",
]

# 可选：response 前缀（引导模型看图）
RESPONSE_PREFIXES = [
    "",
    "根据图片，",
    "图中手势是：",
    "答案是：",
]

# ========== 1. 分析原始数据 ==========
print("=" * 50)
print("分析原始数据...")
print("=" * 50)

responses = []
queries = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        content = item["messages"][0]["content"]
        for c in content:
            if c.get("type") == "text":
                queries.append(c["text"])
            if c.get("type") == "image":
                pass
        responses.append(item["messages"][1]["content"])

print(f"总样本数: {len(responses)}")
print(f"Query 分布: {Counter(queries)}")
print(f"Response 分布: {Counter(responses)}")

# ========== 2. 生成新数据 ==========
print("\n" + "=" * 50)
print("生成改造后的数据...")
print("=" * 50)

random.seed(42)  # 可复现

new_lines = []
name_map = {}  # 旧名 → 新名

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        item = json.loads(line)
        content = item["messages"][0]["content"]
        
        # 提取旧图片路径
        old_img_path = None
        for c in content:
            if c.get("type") == "image":
                old_img_path = c["image"]
                break
        
        if not old_img_path:
            continue
        
        # 生成随机文件名（避免泄露）
        old_name = Path(old_img_path).name
        if old_name not in name_map:
            ext = Path(old_name).suffix
            new_name = f"{uuid.uuid4().hex[:8]}{ext}"
            name_map[old_name] = new_name
            
            # 复制文件到新目录
            src = INPUT_IMG_DIR / old_name
            dst = OUTPUT_IMG_DIR / new_name
            if src.exists():
                shutil.copy(src, dst)
        
        new_img_name = name_map[old_name]
        new_img_path = str(OUTPUT_IMG_DIR / new_img_name).replace("\\", "/")
        
        # 随机选择 query 和 response 前缀
        query = random.choice(QUERY_TEMPLATES)
        prefix = random.choice(RESPONSE_PREFIXES)
        original_response = item["messages"][1]["content"]
        new_response = prefix + original_response if prefix else original_response
        
        # 构造新 item
        new_item = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": new_img_path},
                        {"type": "text", "text": query}
                    ]
                },
                {
                    "role": "assistant",
                    "content": new_response
                }
            ]
        }
        
        new_lines.append(new_item)
        
        if idx % 500 == 0:
            print(f"处理 {idx}/{len(responses)}")

# ========== 3. 保存 ==========
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in new_lines:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ 完成！")
print(f"新数据集: {OUTPUT_JSONL}")
print(f"新图片目录: {OUTPUT_IMG_DIR}")
print(f"总样本: {len(new_lines)}")

# 验证
print("\n" + "=" * 50)
print("验证新数据...")
print("=" * 50)

new_queries = []
new_responses = []
with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        for c in item["messages"][0]["content"]:
            if c.get("type") == "text":
                new_queries.append(c["text"])
        new_responses.append(item["messages"][1]["content"])

print(f"新 Query 分布: {Counter(new_queries)}")
print(f"新 Response 分布: {Counter(new_responses)}")