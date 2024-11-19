from ultralytics import YOLO

# Load a model
model = YOLO("yolov10obb.yaml",task="obb")  # build a new model from YAML

# Train the model
results = model.train(
    data="dota8.yaml",
    epochs=200,
    imgsz=1024,
    batch=16,
    val=True,
    exist_ok=True,
    device="0",
    degrees=180,          # 设置图像旋转的角度范围
    translate=0.1,         # 设置图像平移的范围
    scale=0.5,             # 设置图像缩放的范围
    shear=0.0,             # 设置图像剪切的范围
    flipud=0.5,            # 设置垂直翻转的概率
    fliplr=0.5,            # 设置水平翻转的概率
    hsv_h=0.015,           # 设置色相的调整范围
    hsv_s=0.7,             # 设置饱和度的调整范围
    hsv_v=0.4,             # 设置亮度的调整范围
    mosaic=1.0,            # 启用 Mosaic 数据增强
    mixup=0.1,             # 启用 MixUp 数据增强
    auto_augment='randaugment', # 使用自动增强策略
    erasing=0.4,
    resume=False# 设置随机擦除的概率
)
