# LLM Post-training Projects

基于 **Qwen-VL** 多模态基座的**后训练（Post-training）**项目合集。  
**共享基础**：YOLO 视觉过滤 + Qwen-VL SFT 指令微调；**上层分叉**：DPO 偏好对齐（去冗余）与 RLHF/PPO 场景化语义消歧两条链路。

---

## 技术栈总览

| 层级 | 技术 |
|:---|:---|
| **视觉过滤** | YOLOv8 轻量感知 Agent |
| **基座模型** | Qwen-VL / Qwen3-VL 多模态 SFT（LoRA/QLoRA）|
| **对齐方法** | DPO（直接偏好优化）/ RLHF（Reward Model + PPO）/ GRPO |
| **框架工具** | MS-SWIFT, Transformers, PEFT, TRL, DeepSpeed ZeRO-3 |
| **推理部署** | vLLM, PagedAttention, Gradio |

---

## 共享基座：YOLO + SFT

所有项目基于同一套**视觉感知 + 多模态语言基座**：

```bash
# 1. YOLO 过滤（轻量视觉 Agent）
python src/yolo_detect.py --source input.jpg

# 2. Qwen-VL SFT（共享基座）
python src/sft_qwenvl.py --lora_r 16 --lora_alpha 32
