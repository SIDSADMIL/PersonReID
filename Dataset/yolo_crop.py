import Support
import cv2
import os
import uuid

def detect_crop_person(root_dir,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for dir in os.listdir(root_dir):
        input_dir = root_dir+'/'+dir
        for img in os.listdir(input_dir):
            Frame = cv2.imread(input_dir+'/'+img)
            Bbox , Confidence = Support.detect_persons_yolo(Frame)
            #print('Raw Detections: ',Raw_detections)
            
            for box in Bbox:
                #print('box',box)
                filename=str(uuid.uuid4())
                folder_name = dir
                file_path = output_dir+'/'+folder_name+'_'+filename+'.jpg'
                Support.crop_save_person(Frame, box, file_path)
        print(f'Cropped images from folder : {dir}, to folder : {output_dir}')

