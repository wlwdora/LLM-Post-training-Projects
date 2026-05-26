import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration

class SceneRewardModel(nn.Module):
    """
    输入：<image, scene_context, candidate_answer> → 标量分数
    """
    def __init__(self, model_path, trust_remote_code=True):
        super().__init__()
        # 加载 Qwen-VL 基座，冻结大部分参数
        self.backbone = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
        )
        
        # 冻结 vision tower 和 LLM backbone，只训 score head
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        hidden_size = self.backbone.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化 score head 为较小值，避免初始分数爆炸
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        # 取 LLM 最后一层 hidden state
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 取最后一个有效 token 的 hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, L, H]
        
        # 找到每个序列最后一个非 pad 位置
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_size = last_hidden.size(0)
        pooled = last_hidden[torch.arange(batch_size, device=last_hidden.device), seq_lengths]
        
        score = self.score_head(pooled).squeeze(-1)  # [B]
        return score