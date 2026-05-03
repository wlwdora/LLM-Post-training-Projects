# ️ YOLOv8-Gesture: 基于注意力机制的手势识别系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Framework-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **项目简介**：这是一个从模型改进到端侧部署的完整 AI 实践项目。针对经典的“石头剪刀布”手势识别场景，通过在 YOLOv8 骨干网络中引入 **CBAM 注意力机制**，显著提升了模型在复杂背景下的特征提取能力，并完成了基于 ONNX Runtime 的高性能部署。

---

##  核心亮点

-  **模型架构优化**：在 YOLOv8 的 Backbone 中嵌入 **CBAM (Convolutional Block Attention Module)**，有效抑制环境噪声，聚焦手势关键特征。
-  **极致推理性能**：通过 **ONNX 格式导出与算子融合**，在纯 CPU 环境下推理速度提升 **58%**，完美适配端侧轻量级部署。
- ️ **多平台部署**：基于 **Streamlit** 搭建可视化界面，实现了“上传图片 -> 实时推理 -> 结果渲染”的完整交互闭环。同时可调用手机摄像头，实现实时监控推理与指定图片保存，在ｐｃ与Android端运行均性能良好。

---

##  性能对比实验

我们在相同的测试集上对基线模型与改进模型进行了对比，核心指标如下：

| 模型版本 | mAP50 (精度) | mAP50-95 (综合精度) | CPU 推理耗时 (单帧) |
| :--- | :---: | :---: | :---: |
| YOLOv8 (Baseline) | 0.92 | 0.72 | 39.95 ms |
| **YOLOv8 + CBAM (Ours)** | **0.95** ️ | **0.75** ️ | **25.21 ms** ️ |

>  **结论**：引入注意力机制后，模型不仅精度更高，且经过 ONNX 优化后，推理延迟大幅降低，完全满足实时性要求。

---

## ️ 快速开始

### 1. 环境配置模型准备
确保你的电脑上安装了 Python 3.8 及以上版本，然后克隆本项目并安装依赖：
```bash
git clone https://github.com/你的用户名/yolov8-gesture.git
cd yolov8-gesture
pip install -r requirements.txt

2. 模型准备
本项目已将训练并优化好的 ONNX 模型文件（best.onnx）直接包含在仓库根目录中。克隆项目后即可直接使用，无需额外下载。
