# LLM Post-training Projects

基于 **Qwen-VL** 多模态基座的后训练（Post-training）项目合集。  
**共享基础**：YOLO 视觉过滤 + Qwen-VL SFT 指令微调；**上层分叉**：DPO 偏好对齐（去冗余）与 RLHF/PPO 场景化语义消歧两条链路（RAG模块可选是否需要）。

---

## 项目整体架构

<img width="2774" height="1573" alt="system_architecture" src="https://github.com/user-attachments/assets/1a48ea2c-2f84-4ef0-a218-1333a7d22db9" />

## 项目结构

```text
llm-post-training-projects/
│  README.md
│  requirements.txt
│
├─post_training_framework
│      DPOtrain.py
│      DPO_Datasets_gen.py
│      Model_and_Tokenizer.py
│      PPOtrain.py
│      requirements.txt
│      RMtrain.py
│      SFTtrain.py
│      SFT_Datasets_gen.py
│
├─pre_project
│  ├─asserts
│  │      sft_eval_loss.png
│  │      sft_train_learning_rate.png
│  │      sft_train_loss.png
│  │      yolo_results.png
│  │
│  ├─data
│  │      sample_train.jsonl
│  │
│  └─src
│      ├─SFT
│      │      SFT_data_gen.py
│      │      SFT_qwen3.py
│      │
│      └─Yolo
│              cbam_p3.yaml
│              data.yaml
│              pruned.py
│              Yolo.py
│
├─project-1-dpo-internship
│  ├─assets
│  │      gradio_demo.png
│  │      train_loss.png
│  │      train_rewards_margins.png
│  │
│  ├─data
│  │      sample_train_10.jsonl
│  │
│  └─src
│          DPO_data_gen.py
│          DPO_qwen3.py
│
└─project-2-gesture-rlhf
    ├─assets
    │      ppo_training.png
    │      result.png
    │      rm_training.png
    │
    ├─data
    │      sample_rm_preference_5.jsonl
    │
    └─src
        ├─rag
        │      clip_retrieval.py
        │
        └─rlhf
                ds_config_zero3.json
                ppo_train.py
                rm_model.py
                rm_train.py
```
---

## 技术栈总览

| 层级 | 技术 |
|:---|:---|
| **视觉过滤** | YOLOv8 轻量感知 Agent |
| **基座模型** | Qwen-VL / Qwen3-VL 多模态 SFT（LoRA/QLoRA）|
| **对齐方法** | DPO（直接偏好优化）/ RLHF（Reward Model + PPO） |
| **框架工具** | MS-SWIFT, Transformers, PEFT, TRL, DeepSpeed ZeRO-3 |
| **推理部署** | vLLM, PagedAttention, Gradio |

---

## 共享基座：YOLO + SFT

所有项目基于同一套**视觉感知 + 多模态语言基座**：

```bash
# 1. YOLO 过滤（轻量视觉 Agent）
python src/yolo.py --source input.jpg

# 2. Qwen-VL SFT（共享基座）
python src/sft_qwenvl.py --lora_r 16 --lora_alpha 32
```

- **YOLO**：毫秒级手势检测，拦截无效请求，降低大模型调用成本
- **SFT**：注入视觉-语言对齐能力，使模型具备"描述手势 + 理解场景"的基础表达能力

> SFT 后的合并模型 `merged_sft_qwen_model` 作为两条后训练链路的**公共起点**。

---

## 项目一：基于 DPO 的冗余消除对齐

**路径**：SFT 基座 → **DPO 偏好对齐** → vLLM 推理验证

针对通用大模型在专业软件支持中的**幻觉与冗长生成**问题，构建私有化技术助手。

| 模块 | 说明 |
|:---|:---|
| `project-1-dpo-internship/src/dpo_train.py` | DPO 偏好对齐（β=0.1，KL 约束）|
| `project-1-dpo-internship/src/vllm_inference.py` | vLLM 本地推理（TTFT/TTOT 测试）|

**关键结果**：
- 冗长生成长尾显著抑制
- 本地推理延迟 **<< 1s**

---

## 项目二：基于 RLHF/PPO 的手势语义消歧系统（独立）

**路径**：SFT 基座 → **场景化 Reward Model** → **PPO 策略优化** → vLLM 服务化

在共享 SFT 基座上，针对**同形异义**困境（合影剪刀手 vs 游戏剪刀形态相同、语义冲突），引入 RLHF 做深度语义对齐。

| 模块 | 说明 |
|:---|:---|
| `project-2-gesture-rlhf/src/rlhf/rm_model.py` | 场景化 Reward Model（视觉忠实度 + 场景匹配度）|
| `project-2-gesture-rlhf/src/rlhf/rm_train.py` | RM 训练（Bradley-Terry Loss）|
| `project-2-gesture-rlhf/src/rlhf/ppo_train.py` | PPO 四模型训练（DeepSpeed ZeRO-3）|
| `project-2-gesture-rlhf/src/rag/clip_retrieval.py` | CLIP RAG 兜底检索 |

**关键结果**：
- 同形异义误判率从 **46% → 1.3%**
- 场景语义理解准确率从 **42% → 94%**
- 端到端延迟 **<< 1s**

---

## 数据与模型

由于数据规模与隐私合规，完整数据集与模型权重不直接托管于本仓库。

如有需要可联系1265776769@qq.com

> 各项目 `data/` 目录下已放置 **5-10 条样例数据**，可直接查看格式。

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/wlwdora/llm-post-training-projects.git
cd llm-post-training-projects

# 安装环境（两个项目共用）
pip install -r requirements.txt

# 一：DPO 冗余消除
cd project-1-dpo-internship
python src/sft_qwen3.py        # SFT
python src/dpo_train.py         # DPO
python src/vllm_inference.py  # 推理验证

# 二：RLHF 手势语义系统
cd project-2-gesture-rlhf
python src/rlhf/rm_train.py    # Reward Model
deepspeed --num_gpus=4 src/rlhf/ppo_train.py --deepspeed src/rlhf/ds_config_zero3.json  # PPO
python src/app/gradio_app.py   # UI 部署
```

---

