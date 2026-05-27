# 🚀 LLM 后训练基础框架

这是一个基于 Hugging Face `transformers` 和 `trl` 库构建的轻量级、模块化大语言模型（LLM）后训练框架。旨在为 Qwen-VL 及其他开源模型提供标准化的 **SFT**、**DPO** 和 **PPO** 训练流程。

## ✨ 特性

- **模块化设计**：将模型加载、数据处理和训练逻辑解耦，便于维护和扩展。
- **多算法支持**：内置监督微调、直接偏好优化和近端策略优化三种主流对齐算法。
- **配置灵活**：通过独立的 Python 脚本管理训练参数，易于调试。
- **Qwen-VL 适配**：针对多模态模型进行了专门的 Tokenizer 和 Processor 适配。

## 📂 目录结构

```text
post_training_framework/
├── Model_and_Tokenizer.py    # [核心] 模型与分词器加载工具
├── SFTtrain.py               # 监督微调训练入口
├── DPOtrain.py               # 直接偏好优化训练入口
├── PPOtrain.py               # 近端策略优化训练入口
├── SFT_Datasets_gen.py       # 监督微调数据处理脚本
├── DPO_Datasets_gen.py       # 直接偏好优化数据处理脚本
├── requirements.txt          # 依赖环境文件
└── README.md                 # 项目说明文档
```

## 🛠️ 环境依赖

请确保已安装以下核心库（详见 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `torch`
- `transformers`
- `accelerate`
- `trl`
- `peft`
- `datasets`

## 🏃‍♂️ 快速开始

### 1. 准备模型
确保你的模型路径已在 `Model_and_Tokenizer.py` 中正确配置，或者在训练脚本中指定模型名称（如 `Qwen/Qwen-VL-Chat`）。

### 2. 数据准备
根据你的任务需求，运行对应的数据生成脚本：

```bash
# 生成 SFT 数据集
python SFT_Datasets_gen.py

# 生成 DPO 数据集 (需包含 Prompt/Chosen/Rejected)
python DPO_Datasets_gen.py
```

### 3. 开始训练

**监督微调**
执行全参数微调或 LoRA 微调：
```bash
python SFTtrain.py
```

**直接偏好优化**
基于人类偏好数据对齐模型：
```bash
python DPOtrain.py
```

**近端策略优化**
基于奖励模型的强化学习训练：
```bash
python PPOtrain.py
```

## ⚙️ 核心模块说明

### 🧩 Model_and_Tokenizer.py
该模块封装了模型加载逻辑，包括：
- **AutoConfig**: 模型配置加载。
- **AutoModelForCausalLM**: 支持 4-bit/8-bit 量化加载以节省显存。
- **PeftModel**: 集成 LoRA 配置，支持高效参数微调。

### 📊 数据处理
- **SFT**: 将指令数据转换为 chatml 或其他模型所需的对话模板格式。
- **DPO**: 构建包含 `prompt`, `chosen`, `rejected` 字段的偏好数据集。

```
