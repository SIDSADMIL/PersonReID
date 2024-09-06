import os
import shutil
import math
import uuid
import cv2

Pdirname = 'Person'
Cdirname = 'Camera'
DatasetType='ivisreid'#'market1501'
Traindirname='bounding_box_train' if DatasetType == 'market1501' else 'Train'
Gallerydirname='bounding_box_test' if DatasetType == 'market1501' else 'Gallery'
Querydirname='query' if DatasetType == 'market1501' else 'Query'
GalleryQuery_Train_Split_Ration = '50:50'

root_dir='C:\IVIS\PersonReID\Dataset\Backup'
dataset_dir = 'C:\IVIS\PersonReID\Dataset\Backup\Test'

"""
1. "bounding_box_test" file. This file contains 19732 images for testing. 
2. "bounding_box_train" file. This file contains 12936 images for training.
3. "query" file. It contains 3368 query images. Search is performed in the "bounding_box_test" file.


0001_c1s1_001051_00.jpg . If we split by the character “_” we will have 4 parts as follows.

0001 — ID of the person in the image
c1 — Number of the camera from which the image was captured
s1 — Number of the sequence
001051 — The number of the frame in the video sequence from which the image was extracted
00 — Number of the bounding box detected in the frame.
"""

# Define a function to assign names to images
def assign_names(camera_dir, person_id, camera_id):
    # get all the files in the folder
    image_files = os.listdir(camera_dir)
    i=1
    for image_file in image_files:
        # for each file get the file path
        src_img_path = camera_dir+'/'+image_file
        # for each file split and get the name and extension
        filename, extension = os.path.splitext(image_file)
        #print(camera_dir, person_id, camera_id)
        # for each file rebuild the path with proper file name (PID_CID_XXXX.extension)
        name = f"{int(person_id):04d}_c{camera_id}s1_{i:06d}_00{extension}"
        dst_img_path = camera_dir+'/'+name
        # rename the file with new name in the same folder
        shutil.move(src_img_path, dst_img_path)
        i+=1

def copy_images(input_folder, person_folder, path_to_copy, copy_to_query):
    print('Copying files..................................')
    # for each person get the folder path
    person_path = input_folder+'/'+person_folder
    if os.path.isdir(person_path):
        # if person folder exisits
        for camera_folder in os.listdir(person_path):
            # for each cam folder get the folder path
            camera_path = person_path+'/'+camera_folder
            if os.path.isdir(camera_path):
                # if the cam folder exists
                imgno = 0
                for img in os.listdir(camera_path):
                    image_path = camera_path+'/'+img
                    if os.path.isfile(image_path):
                        # by default copy all to gallery
                        shutil.copy(image_path, path_to_copy)
                        if (copy_to_query and (imgno == 0)):
                            dir_split = (path_to_copy.split('/'))[:-1]
                            #print(path_to_copy, dir_split)
                            delim = '/'
                            dir_path = delim.join(folder for folder in dir_split)
                            q_path = dir_path+'/'+Querydirname
                            #print(dir_path,q_path)
                            shutil.copy(image_path, q_path)
                    imgno+=1

def SetupQuery(Qpath):
    print('Setting up query folder .................')
    print('Qpath'+Qpath)
    for pdir in os.listdir(Qpath):
        pdirpath=Qpath+'/'+pdir
        for cdir in os.listdir(pdirpath):
            cdirpath=pdirpath+'/'+cdir
            for i, file in enumerate(os.listdir(cdirpath)):
                filepath = cdirpath+'/'+file
                if i!=0:
                    os.remove(filepath)
                    
def copy_folders(input_folder, person_folder, path_to_copy, copy_to_query):
    print('Copying folders.....................')
    person_path = input_folder+'/'+person_folder
    path_to_copy = path_to_copy+'/'+person_folder
    print('copying: '+person_path+' to: '+path_to_copy)
    if os.path.isdir(person_path):
        shutil.copytree(person_path, path_to_copy, dirs_exist_ok=True)
        if copy_to_query:
            dir_split = (path_to_copy.split('/'))[:-2]
            #print(path_to_copy, dir_split)
            delim = '/'
            dir_path = delim.join(folder for folder in dir_split)
            q_path = dir_path+'/'+Querydirname
            q_path_Curr_person = q_path+'/'+person_folder
            #print(dir_path,q_path)
            shutil.copytree(person_path, q_path_Curr_person, dirs_exist_ok=True)
            SetupQuery(q_path)

# Copy / build new data set matching market 1501 / coustom dataset
def copy_dataset(input_folder, output_folder):
    print('Copying dataset....................')
    # Create output folder if it doesn't exist
    #if not os.path.exists(output_folder):
    os.makedirs(output_folder,exist_ok=True)
    # Create train, query, and gallery folders
    train_folder = output_folder+'/'+Traindirname
    query_folder = output_folder+'/'+Querydirname
    gallery_folder = output_folder+'/'+Gallerydirname
    os.makedirs(train_folder,exist_ok=True)
    os.makedirs(query_folder,exist_ok=True)
    os.makedirs(gallery_folder,exist_ok=True)

    TotalPdirs = len(os.listdir(input_folder))
    Gal_Q_Count = math.ceil(TotalPdirs * (int(GalleryQuery_Train_Split_Ration.split(':')[0])/100))
    dirno = 1

    # Copy images to train, query and gallery folders based on the split ratio
    for person_folder in os.listdir(input_folder):
        if dirno <= Gal_Q_Count:
            if DatasetType == 'market1501':
                # copy persons into gallery and query
                copy_images(input_folder, person_folder, gallery_folder, True)
            else:
                # move folders
                copy_folders(input_folder, person_folder, gallery_folder, True)
        else:
            if DatasetType == 'market1501':
                # copy persons into train
                copy_images(input_folder, person_folder, train_folder, False)
            else:
                # move folders
                copy_folders(input_folder, person_folder, train_folder, False)
        dirno+=1

def Rename_Raw_Data_Set(dataset_dir):
    pid=115
    for dir in (os.listdir(dataset_dir)):
        curr_dir = dataset_dir+'/'+str(dir)
        new_dir = dataset_dir+'/'+str(Pdirname+str(pid))
        os.rename(curr_dir,new_dir)
        cid=1
        for subdir in (os.listdir(new_dir)):
            curr_subdir = new_dir+'/'+str(subdir)
            new_subdir = new_dir+'/'+Cdirname+str(cid)
            os.rename(curr_subdir,new_subdir)
            cid+=1
        pid+=1
    print('Renaming folders done.........')

def Setup_Data_Set(dataset_dir,output_dir):
    print('Setting up the dataset.........')
    Rename_Raw_Data_Set(dataset_dir)
    Person_IDs = []
    Cam_IDs = []
    if DatasetType == 'market1501':
        # Traverse the dataset directory and rename all files
        for person_dir in (sorted(os.listdir(dataset_dir))):
            # for each person get the path to the folder
            person_dir = dataset_dir+'/'+str(person_dir)
            if os.path.isdir(person_dir):
                # get the person id
                person_id = person_dir.split(Pdirname)[1]
                Person_IDs.append(person_id)
                for camera_dir in (sorted(os.listdir(person_dir))):
                    # get the cam id
                    camera_id = camera_dir.split(Cdirname)[1]
                    Cam_IDs.append(camera_id)
                    # for each camera get the path to the folder
                    camera_dir = person_dir+'/'+str(camera_dir)
                    #print(person_id+':'+ camera_id)
                    assign_names(camera_dir, person_id, camera_id)
        print('Renaming files done.............')
    # Copy dataset
    copy_dataset(dataset_dir, output_dir)

# Define the dataset directory
#'DataSet_Input'

output_folder = 'market1501/Market-1501-v15.09.15' if DatasetType == 'market1501' else 'iVISREid_REid_Dataset'
output_folder = root_dir+'/'+output_folder if root_dir else output_folder

Setup_Data_Set(dataset_dir,output_folder)

