import cv2
import tensorflow as tf
import numpy as np
import random 
#import deep_sort
import os
import time

import uuid

RAWdetections =[]
detector = tf.saved_model.load("model_folder")
path_to_save='C:/MotivityLabs/AiMl_Practice/DeepSort/Detected_Cropped_DS/'

# Define a function to process each video frame
def detect_objects(frame):
  
  tempFrame=frame
  frame = np.expand_dims(frame, axis=0)
  # Get detections from the model
  detections = detector(frame)
  # Retrieve relevant information from detections dictionary
  boxes = detections['detection_boxes'][0]
  print("Box shape",boxes.shape)
  im_height, im_width,_ = tempFrame.shape
  boxes_list = [None for i in range(boxes.shape[1])]
  for i in range(boxes.shape[1]):
      boxes_list[i] = (int(boxes[i,0] * im_height),int(boxes[i,1]*im_width),int(boxes[i,2]*im_height),int(boxes[i,3]*im_width))
  boxes=boxes_list
  classes = detections['detection_classes'][0].numpy().astype(np.int32)
  scores = detections['detection_scores'][0].numpy()
  # Draw bounding boxes and labels on the frame

  for i in range(len(boxes)):
    if classes[i] == 1 and scores[i] > 0.9:  # Set a minimum confidence threshold
      box = boxes[i]
      y_min, x_min, y_max, x_max = box
      RAWdetections.append([x_min,y_min,x_max,y_max])

  return RAWdetections

def crop_save_image(imgFrame,box,path_to_save):
    os.makedirs(path_to_save,exist_ok=True)
    cropped_image=imgFrame[int(box[1]):int(box[3]),int(box[0]):int(box[2])] # type: ignore
    file_guid=uuid.uuid4()
    cv2.imshow('frame',imgFrame)
    cv2.imshow('Cropped Image',cropped_image)
    time.sleep(60)
    cv2.imwrite(path_to_save+'/'+str(file_guid)+'.jpg',cropped_image)

def detect_crop_image(input_folder_path,path_to_save):
    image_files=os.listdir(input_folder_path)
    for image in image_files:
        img_path=input_folder_path+'/'+image
        print(img_path)
        img=cv2.imread(img_path)
        DetectionData=detect_objects(img)
        for j in DetectionData:
            box=j
            crop_save_image(img,box,path_to_save)
    return path_to_save

#detect_crop_image('C:\MotivityLabs\AiMl_Practice\DeepSort\TestImages',path_to_save)
