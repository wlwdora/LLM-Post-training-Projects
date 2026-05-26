import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLProcessor
from accelerate import Accelerator
from rm_model import SceneRewardModel

# ========== 配置 ==========
MODEL_PATH = "../sft/merged_sft_qwen_model"  # SFT 后的基座
OUTPUT_DIR = "./output/rm"
BATCH_SIZE = 2
LR = 1e-5
EPOCHS = 3
MAX_LENGTH = 512
# =======================

class PreferenceDataset(Dataset):
    """场景化偏好数据集：同一张图，不同场景下的 chosen vs rejected"""
    def __init__(self, data_path, processor):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        scene = item["scene_context"]  # 如"朋友聚会"
        question = item["question"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # 构造带场景提示的文本
        prompt = f"场景：{scene}\n问题：{question}\n回答："
        
        # chosen 样本
        chosen_text = prompt + chosen
        # rejected 样本
        rejected_text = prompt + rejected
        
        return {
            "image": image_path,
            "chosen_text": chosen_text,
            "rejected_text": rejected_text,
            "scene": scene,
        }

def collate_fn(batch, processor):
    images = [b["image"] for b in batch]
    chosen_texts = [b["chosen_text"] for b in batch]
    rejected_texts = [b["rejected_text"] for b in batch]
    
    # 处理 chosen
    chosen_inputs = processor(
        text=chosen_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    
    # 处理 rejected
    rejected_inputs = processor(
        text=rejected_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    
    return {
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "chosen_pixel_values": chosen_inputs.get("pixel_values"),
        "chosen_image_grid_thw": chosen_inputs.get("image_grid_thw"),
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"],
        "rejected_pixel_values": rejected_inputs.get("pixel_values"),
        "rejected_image_grid_thw": rejected_inputs.get("image_grid_thw"),
    }

def bt_loss(chosen_scores, rejected_scores):
    """Bradley-Terry pairwise loss"""
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()

def main():
    accelerator = Accelerator(mixed_precision="bf16")
    
    processor = Qwen2VLProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = SceneRewardModel(MODEL_PATH)
    
    # 只优化 score head
    optimizer = torch.optim.AdamW(model.score_head.parameters(), lr=LR, weight_decay=0.01)
    
    dataset = PreferenceDataset("./data/rm_preference.jsonl", processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           collate_fn=lambda x: collate_fn(x, processor))
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    model.train()
    global_step = 0
    
    for epoch in range(EPOCHS):
        for batch in dataloader:
            # Forward chosen
            chosen_scores = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                pixel_values=batch.get("chosen_pixel_values"),
                image_grid_thw=batch.get("chosen_image_grid_thw"),
            )
            
            # Forward rejected
            rejected_scores = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                pixel_values=batch.get("rejected_pixel_values"),
                image_grid_thw=batch.get("rejected_image_grid_thw"),
            )
            
            loss = bt_loss(chosen_scores, rejected_scores)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.score_head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if global_step % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}, "
                      f"Chosen: {chosen_scores.mean().item():.2f}, "
                      f"Rejected: {rejected_scores.mean().item():.2f}")
            
            global_step += 1
    
    # 保存
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), 
                   os.path.join(OUTPUT_DIR, "rm_model.pt"))
        print(f"Reward Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()