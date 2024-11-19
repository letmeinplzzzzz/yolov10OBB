from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a custom model

# Validate the model
metrics = model.val(data="dotav1.yaml",save_json=True,split="test",device=1,conf=0.001)  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category