'''

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
#imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
imgs = ['image.jpg']  # batch of images

# Inference
results = model(imgs)
# Results
results.print()
#results.show()
#results.save()  # or .show()

print(results.xyxy[0])  # img1 predictions (tensor)
print(results.pandas().xyxy[0]) # img1 predictions (pandas)
'''
import Support
import cv2
import os
import uuid

input_dir ='input'
output_dir ='output'

os.makedirs(output_dir, exist_ok=True)

for img in os.listdir(input_dir):
    Frame = cv2.imread(input_dir+'/'+img)
    Bboxes = Support.detect_persons_yolo(Frame)
    for box in Bboxes:
        filename=str(uuid.uuid4())
        folder_name = img.split('_')[0]
        file_path = output_dir+'/'+folder_name+'_'+filename+'.jpg'
        Support.crop_save_person(Frame, box, file_path)
