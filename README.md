# ✌️ YOLOv8-Gesture: 基于注意力机制的手势识别系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Framework-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **项目简介**：这是一个从模型改进到端侧部署的完整 AI 实践项目。针对经典的“石头剪刀布”手势识别场景，通过在 YOLOv8 骨干网络中引入 **CBAM 注意力机制**，显著提升了模型在复杂背景下的特征提取能力，并完成了基于 ONNX Runtime 的高性能部署。

---

## ✨ 核心亮点

- 🧠 **模型架构优化**：在 YOLOv8 的 Neck 中嵌入 **CBAM (Convolutional Block Attention Module)**，有效抑制环境噪声，聚焦手势关键特征。
- ⚡ **极致推理性能**：通过 **ONNX 格式导出与算子融合**，在纯 CPU 环境下推理速度提升 **58%**，完美适配端侧轻量级部署。
- 🖥️ **交互式 Web 应用**：基于 **Gradio** 搭建可视化界面，实现了“上传图片 -> 实时推理 -> 结果渲染”的完整交互闭环。

---

## 📊 性能对比实验

我们在相同的测试集上对基线模型与改进模型进行了对比，核心指标如下：

| 模型版本 | mAP50 (精度) | mAP50-95 (综合精度) | CPU 推理耗时 (单帧) |
| :--- | :---: | :---: | :---: |
| YOLOv8 (Baseline) | 0.92 | 0.72 | 39.95 ms |
| **YOLOv8 + CBAM (Ours)** | **0.95** ⬆️ | **0.75** ⬆️ | **25.21 ms** ⬇️ |

> **结论**：引入注意力机制后，模型不仅精度更高，且经过 ONNX 优化后，推理延迟大幅降低，完全满足实时性要求。

---

### 2. 模型准备
请将训练好的 ONNX 模型文件（命名为 `best.onnx`）放置于项目根目录下。
> 注：由于 GitHub 文件大小限制，模型文件需自行训练或从下方链接获取。

### 3. 启动应用
在终端运行以下命令启动 Web 交互界面：

```bash
python app.py
```

启动成功后，浏览器将自动弹出（或访问 `http://localhost:7860`），上传图片即可体验实时识别。

---

## 📁 项目结构

```
YOLOv8-Hand-Gesture-Recognition/
├── app.py                  # Gradio 交互式前端 & ONNX 推理核心逻辑
├── requirements.txt        # 项目依赖库清单
├── best.onnx               # 导出的轻量化推理模型
├── cbam_p3.yaml            # 带 CBAM 的模型配置文件
├── data.yaml               # 数据集配置
├── assets/                 # 演示素材（截图、GIF等）
└── README.md               # 项目说明文档
```

---

## 🖼️ 效果展示

*(建议在这里放一张你的识别截图或 GIF 动图)*

---

## 📝 技术博客

[项目完整解析](https://你的博客链接)

---

## 📄 许可证

本项目仅供学习与技术交流使用，遵循 MIT 开源协议。
```
