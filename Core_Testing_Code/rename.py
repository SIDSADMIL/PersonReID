import os
import shutil
import math
import uuid
import cv2

inputdir = 'Raw_Data'
output = 'Output'

def Rename_Raw_Data_Set(inputdir):
    for i,file in enumerate((os.listdir(inputdir))):
        curr_file = inputdir+'/'+str(file)
        new_img_name=str(i)+'.jpg'
        src=curr_file
        dest = inputdir+'/'+new_img_name
        os.rename(src,dest)
        #shutil.copy(new_img_path, dest)
    print('done.........')
    
    
Rename_Raw_Data_Set('TempImage/allcropped')