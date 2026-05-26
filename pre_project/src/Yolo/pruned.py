import time
from ultralytics import YOLO
import os
import torch
import glob

if __name__ == '__main__':
    # 强制使用 CPU
    torch.cuda.is_available = lambda: False
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # 切换到数据集目录
    os.chdir("D:/datasets")
    
    # 1. 加载模型 (强制 CPU)
    print("正在加载模型...")
    original_model = YOLO("runs/detect/runs/cbam_finetune/exp-5/weights/best.pt")
    original_model.model.to('cpu')
    
    onnx_model = YOLO("runs/detect/runs/cbam_finetune/exp-5/weights/best.onnx")
    
    # 2. 验证精度 (mAP)
    print("\n========== 正在验证原始模型精度 ==========")
    metrics_original = original_model.val(data="data.yaml")
    print(f"原始模型 mAP50: {metrics_original.box.map50:.4f}")
    print(f"原始模型 mAP50-95: {metrics_original.box.map:.4f}")
    
    print("\n========== 正在验证 ONNX 模型精度 ==========")
    metrics_int8 = onnx_model.val(data="data.yaml")
    print(f"ONNX 模型 mAP50: {metrics_int8.box.map50:.4f}")
    print(f"ONNX 模型 mAP50-95: {metrics_int8.box.map:.4f}")
    
    # 3. 测试推理速度
    test_images = glob.glob("D:/datasets/test.jpg")
    if not test_images:
        print("未找到测试图片")
        exit()
    test_img = test_images[0]
    print(f"\n使用测试图片: {test_img}")
    
    print("\n========== 正在测试推理速度 (CPU) ==========")
    
    # 预热
    for _ in range(10):
        original_model.predict(test_img, verbose=False)
        onnx_model.predict(test_img, verbose=False)
    
    # 测试原始模型速度
    start = time.time()
    for _ in range(100):
        original_model.predict(test_img, verbose=False)
    original_time = (time.time() - start) / 100
    
    # 测试 ONNX 模型速度
    start = time.time()
    for _ in range(100):
        onnx_model.predict(test_img, verbose=False)
    onnx_time = (time.time() - start) / 100
    
    print(f"原始模型 (PyTorch) 平均推理时间: {original_time*1000:.2f} ms")
    print(f"ONNX 模型平均推理时间: {onnx_time*1000:.2f} ms")
    
    if original_time > 0 and onnx_time > 0:
        print(f"速度提升倍数: {original_time / onnx_time:.2f}x")
    
    # 4. 单张图片推理演示
    print("\n========== 单张图片推理演示 ==========")
    results = original_model(test_img)
    results[0].show()