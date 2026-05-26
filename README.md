# LLM Post-training Projects

基于 **Qwen-VL** 多模态基座的**后训练（Post-training）**项目合集。  
**共享基础**：YOLO 视觉过滤 + Qwen-VL SFT 指令微调；**上层分叉**：DPO 偏好对齐（去冗余）与 RLHF/PPO 场景化语义消歧两条链路。

---

## 项目整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                     共享视觉-语言基座                          │
│  ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │  YOLO Agent  │    │      Qwen-VL SFT (LoRA/QLoRA)    │  │
│  │  毫秒级过滤  │ →  │   视觉-语言对齐 · 基础语义表达   │  │
│  │  降低无效调用 │    │   merged_sft_qwen_model (公共起点) │  │
│  └─────────────┘    └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
│
┌───────────────────┴───────────────────┐
▼                                       ▼
┌─────────────────────┐              ┌─────────────────────────────┐
│   项目一：DPO 对齐   │              │   项目二：RLHF/PPO 语义消歧  │
│   (实习 · 已跑通)    │              │   (独立 · 方案已验证)        │
├─────────────────────┤              ├─────────────────────────────┤
│ 目标：抑制幻觉与冗长 │              │ 目标：同形异义场景化理解       │
│                     │              │                             │
│ SFT 基座            │              │ SFT 基座                    │
│   ↓                 │              │   ↓                         │
│ DPO (β=0.1)         │              │ 场景化 Reward Model          │
│   ↓                 │              │   ↓                         │
│ vLLM 推理验证        │              │ PPO (ZeRO-3 分布式)         │
│                     │              │   ↓                         │
│ 结果：采纳率 +25%    │              │ vLLM 服务化部署              │
│       延迟 << 1s    │              │                             │
│                     │              │ 结果：误判率 46% → 1.3%      │
└─────────────────────┘              │       准确率 42% → 94%     │
                                     └─────────────────────────────┘

- **YOLO**：毫秒级手势检测，拦截无效请求，降低大模型调用成本
- **SFT**：注入视觉-语言对齐能力，使模型具备"描述手势 + 理解场景"的基础表达能力

> SFT 后的合并模型 `merged_sft_qwen_model` 作为两条后训练链路的**公共起点**。

---

## 项目一：基于 DPO 的私有化助手对齐（实习）

**路径**：SFT 基座 → **DPO 偏好对齐** → vLLM 推理验证

针对通用大模型在专业软件支持中的**幻觉与冗长生成**问题，构建私有化技术助手。

| 模块 | 说明 |
|:---|:---|
| `project-1-dpo-internship/src/sft_data_gen.py` | 领域指令数据构造（含私有 API 描述）|
| `project-1-dpo-internship/src/sft_qwen3.py` | SFT 训练（MS-SWIFT + LoRA）|
| `project-1-dpo-internship/src/dpo_train.py` | DPO 偏好对齐（β=0.1，KL 约束）|
| `project-1-dpo-internship/src/merge_lora.py` | 权重合并 |
| `project-1-dpo-internship/src/vllm_inference.py` | vLLM 本地推理（TTFT/TTOT 测试）|

**关键结果**：
- DPO 后回答采纳率提升 **25%**
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
| `project-2-gesture-rlhf/src/app/gradio_app.py` | Gradio UI 部署 |

**关键结果**：
- 同形异义误判率从 **46% → 1.3%**
- 场景语义理解准确率从 **42% → 94%**
- 端到端延迟 **<< 1s**

---

## 数据与模型

由于数据规模与隐私合规，完整数据集与模型权重不直接托管于本仓库。

| 资源 | 说明 | 获取 |
|:---|:---|:---|
| `train_sft.jsonl` | 共享 SFT 指令数据 | [百度网盘](xxx) |
| `train_dpo.jsonl` | DPO 成对偏好数据 | [百度网盘](xxx) |
| `train_rm.jsonl` | RLHF 场景化偏好数据 | [百度网盘](xxx) |
| `images_v1.tar.gz` | 手势图片数据集（脱敏）| [百度网盘](xxx) |
| `merged_sft_qwen_model` | 共享 SFT 基座 | [HuggingFace](xxx) |
| `dpo_lora_adapter` | DPO 对齐后 LoRA 权重 | [HuggingFace](xxx) |
| `rm_model.pt` | 场景化 Reward Model | [阿里云盘](xxx) |

> 各项目 `data/` 目录下已放置 **5-10 条样例数据**，可直接查看格式。

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourname/llm-post-training-projects.git
cd llm-post-training-projects

# 安装环境（两个项目共用）
pip install -r requirements.txt

# 项目一：DPO 私有化助手
cd project-1-dpo-internship
python src/sft_qwen3.py        # SFT
python src/dpo_train.py         # DPO
python src/vllm_inference.py  # 推理验证

# 项目二：RLHF 手势语义系统
cd project-2-gesture-rlhf
python src/rlhf/rm_train.py    # Reward Model
deepspeed --num_gpus=4 src/rlhf/ppo_train.py --deepspeed src/rlhf/ds_config_zero3.json  # PPO
python src/app/gradio_app.py   # UI 部署
```

---

## 联系方式

- 可长期实习（6 个月以上）
- Email: your_email@whu.edu.cn
```
