import os
import time
import numpy as np
import cv2
import torch
import Support
from torchreid.reid.utils import *
from tqdm import tqdm


#tqdm(


metric = 'euclidean'#cosine(-0.0),euclidean
dir_path = 'TempImage/allcropped'
img1_path = 'TempImg.jpg'
#img2_path = 'positive.jpg'
#img3_path='test.jpg'
distances=[]
dvals = []
files = []
FMI1=Support.Extract_Feature_Maps(img1_path)
print('anchor features: ',FMI1)
'''
FMI3=Support.Extract_Feature_Maps(img2_path)
print('positive features: ',FMI3)
FMI4=Support.Extract_Feature_Maps(img3_path)
print('negative features: ',FMI4)
'''

for Ifile in tqdm(os.listdir(dir_path)):
    img2_path = dir_path+'/'+Ifile
    files.append(Ifile)
    FMI2=Support.Extract_Feature_Maps(img2_path)
    distance=Support.compute_distance_matrix(FMI1,FMI2,metric=metric)
    distances.append(distance)
    v=round(((distance.tolist())[0])[0], 2) 
    print(v)
    dvals.append(v)
    

i=0
for file in files:
    print(f"File: {file} , distance: {dvals[i]}")
    i+=1

mean = torch.mean(torch.stack(distances))
print("mean: ",mean)
mean = sum(dvals)/len(dvals)
print("mean: ",mean)

'''
tempdvals = dvals

dvals.sort()
print(dvals)

i=0
for val in tempdvals:
    if val == dvals[0]:
        print(f"Best Matched File: {files[i]} , distance: {val}")
    i+=1
'''

