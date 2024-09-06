import os
import shutil
import math
import uuid
import cv2

input = 'Raw_Data'
croppedinput = 'C:/IVIS/PersonReID/Dataset/person'
allimgoutput = 'Output'
rawdatasetoutput = 'C:/IVIS/PersonReID/Dataset/Raw_Data'
no_of_splits = 4
 
def Rename_Copy_Raw_Data_Set(input,allimgoutput):
    os.makedirs(allimgoutput,exist_ok=True)
    for dir in (os.listdir(input)):
        curr_dir = input+'/'+str(dir)
        for img in (os.listdir(curr_dir)):
            img_path = curr_dir+'/'+img
            new_img_name=dir+'_'+img
            new_img_path = curr_dir+'/'+new_img_name
            dest = allimgoutput+'/'+new_img_name
            os.rename(img_path,new_img_path)
            shutil.copy(new_img_path, dest)
    print('done.........')
 
#Rename_Raw_Data_Set(input,allimgoutput)

def Parse_CroppedImgs_To_Data_Set(croppedinput,rawdatasetoutput):
    os.makedirs(rawdatasetoutput,exist_ok=True)
    for croppedimg in (os.listdir(croppedinput)):
        curr_img_path = croppedinput+'/'+str(croppedimg)
        folder_name = croppedimg.split('_')[0]
        folder_path = rawdatasetoutput+'/person'+folder_name
        os.makedirs(folder_path,exist_ok=True)
        dest = folder_path+'/'+croppedimg
        shutil.copy(curr_img_path, dest)
    print('done.........')
    
#Parse_CroppedImgs_To_Data_Set(croppedinput,rawdatasetoutput)

def move_images_into_subdirs(dir_path,no_of_splits):
    if os.path.isdir(dir_path):
        camera_folders = []
        for split in range(no_of_splits):
            sub_dir_name = f"camera{split+1}"
            #each cam folder name
            camera_folders.append(sub_dir_name)
            
        #each img path
        image_paths = [os.path.join(dir_path, img) for img in os.listdir(dir_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        num_images = len(image_paths)
        print('imgs :',num_images)
        images_per_camera = int(num_images / no_of_splits)
        print('imgs per dir:',images_per_camera)
        remaining_images = num_images % no_of_splits
        print('remaining imgs:',remaining_images)
        start_index = 0
        end_index = images_per_camera   
        
        for i, camera_folder in enumerate(camera_folders):
            camera_path = os.path.join(dir_path, camera_folder)
            os.makedirs(camera_path, exist_ok=True)
            if i != 0: 
                start_index = start_index + images_per_camera
                # For the last camera folder, include remaining images
                if i == (no_of_splits-1):
                    end_index = start_index + images_per_camera + remaining_images
                else:
                    end_index = start_index + images_per_camera

            camera_images = image_paths[start_index:end_index]
            for img_path in camera_images:
                shutil.move(img_path, camera_path)
'''
for dir in os.listdir('C:/IVIS/PersonReID/Dataset/Backup/2'):
    curr_dir_path = 'C:/IVIS/PersonReID/Dataset/Backup/2/'+str(dir)
    move_images_into_subdirs(curr_dir_path,no_of_splits)
'''
    
    
curr_dir_path = 'C:/IVIS/PersonReID/Dataset/data'
move_images_into_subdirs(curr_dir_path,no_of_splits)