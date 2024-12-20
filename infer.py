import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("single/train/weights/best.pt")

# from PIL
im1 = Image.open("/home/zeyang/code/datasets/DOTAv1-split_500/images/test/P0006__1024__0___0.jpg")
results = model.predict(source=im1, save=True,show_labels=False,show_conf=False)  # save plotted images

