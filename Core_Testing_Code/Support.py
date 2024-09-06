from torchreid.reid.utils import FeatureExtractor
#from torchreid.reid.utils.rerank import *
#from torchreid.reid.metrics.distance import *
from torch.nn import functional as F
from torchsummary import summary
import cv2
import tensorflow as tf
import numpy as np
import random
import os
import time
import uuid
import torch

QIheight = 256
QIwidth = 128


Feature_Map_Buffer = {}
PCounter = 0
PReID_Thershold = 20
PDetec_Threshold = 0.5
PReID_Tolorance = 0
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
#feature_extractor_model_path = 'model1.pth'
#feature_extractor_model_path = 'market1501_AGW.pth'
feature_extractor_model_path='89EModel.pth'
# ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512', 'se_resnet50', 'se_resnet50_fc512', 'se_resnet101', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512', 'inceptionresnetv2', 'inceptionv4', 'xception', 'resnet50_ibn_a', 'resnet50_ibn_b', 'nasnsetmobile', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'shufflenet', 'squeezenet1_0', 'squeezenet1_0_fc512', 'squeezenet1_1', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'mudeep', 'resnet50mid', 'hacnn', 'pcb_p6', 'pcb_p4', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25']
extractor = FeatureExtractor(model_name='osnet_ain_x1_0',model_path=feature_extractor_model_path, device=Device) #osnet_x1_0
summary(extractor.model, input_size=(3, 256, 128))
metric = 'euclidean'#'euclidean'
person_detection_model_path = 'model_folder'
detector = tf.saved_model.load(person_detection_model_path)
# yolo Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    #print('calculating euclidean distance')
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    return distmat

def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    #print('calculating cosine distance')
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat

def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input1.dim())
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input2.dim())
    assert input1.size(1) == input2.size(1)
    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric))
    return distmat

def crop_save_person(imgFrame,box,path_to_save):
    y_min, x_min, y_max, x_max = box
    
    dir_split = (path_to_save.split('/'))[:-1]
    delim = '/'
    dir_path = delim.join( folder for folder in dir_split)
    allcropped_path = dir_path+'/allcropped'
    #os.makedirs(dir_path, exist_ok=True)
    os.makedirs(allcropped_path, exist_ok=True)
    
    #cropped_image=imgFrame[int(box[1]):int(box[3]),int(box[0]):int(box[2])] # type: ignore
    cropped_image = imgFrame[int(y_min):int(y_max), int(x_min):int(x_max)]  # type: ignore
    #QIheight = 256
    #QIwidth = 128
    cropped_image = cv2.resize(cropped_image, (QIwidth, QIheight))
    
    #cv2.imshow('frame',imgFrame)
    #cv2.imshow('Cropped Image',cropped_image)
    cv2.imwrite(path_to_save,cropped_image)
    filenm=str(uuid.uuid4())
    cv2.imwrite(allcropped_path+'/'+filenm+'.jpg',cropped_image)
    #dirname=str(uuid.uuid4())
    #os.makedirs(dirname, exist_ok=True)
    #filename=uuid.uuid4()
    #cv2.imwrite(dirname+'/'+str(filename)+'.jpg', cropped_image)
    #shutil.copy(path_to_save, dest_path)
    # Rename the copied file
    #new_path = f"{dest_path}/{new_name}"
    #shutil.move(f"{dest_path}/{src_path}", new_path)

def detect_persons(frame):
    RAWdetections = []
    confidences = []
    tempFrame=frame
    frame = np.expand_dims(frame, axis=0)
    # Get detections from the model
    detections = detector(frame)
    # Retrieve relevant information from detections dictionary
    boxes = detections['detection_boxes'][0]
    im_height, im_width,_ = tempFrame.shape
    boxes_list = [None for i in range(boxes.shape[1])]
    for i in range(boxes.shape[1]):
        boxes_list[i] = (int(boxes[i,0] * im_height),int(boxes[i,1]*im_width),int(boxes[i,2]*im_height),int(boxes[i,3]*im_width))
    boxes=boxes_list
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > PDetec_Threshold:  # Set a minimum confidence threshold
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            #RAWdetections.append([x_min,y_min,x_max,y_max])
            RAWdetections.append([y_min, x_min, y_max, x_max])
            confidences.append(scores)
    return RAWdetections, confidences

def detect_persons_yolo(frame):
    RAWdetections = []
    confidences = []
    # Inference
    results = model(frame)
    # Results
    results.print()
    # results.show()
    # results.save()  # or .show()
    #print(results.xyxy[0])  # img1 predictions (tensor)
    #print(results.pandas().xyxy[0])  # img1 predictions (pandas)
    #print(np.array(results.xyxy[0]))
    for detection in np.array(results.xyxy[0]): #results.pandas().xyxy[0] :
        detection_object = detection[5]
        confidence = detection[4]
        #if detection_object == 0:
        if detection_object == 0 and confidence > PDetec_Threshold :
            y_min, x_min, y_max, x_max = detection[1], detection[0], detection[3], detection[2]
            RAWdetections.append([y_min, x_min, y_max, x_max])
            confidences.append(confidence)
    return RAWdetections , confidences


def Extract_Feature_Maps(imagepath):
    features = extractor(imagepath)
    #print("Extracted feature maps from :",imagepath)
    #print("Feature map , size:",features, len(features))
    return features

def Flush_FBuffer():
    global Feature_Map_Buffer
    global PCounter
    Feature_Map_Buffer = {}
    PCounter = 0

def Add_To_FBuffer(PersonID, Features):
    global Feature_Map_Buffer
    global PCounter
    Feature_Map_Buffer[PersonID] = Features

def Remove_From_FBuffer(PersonID):
    global Feature_Map_Buffer
    global PCounter
    del Feature_Map_Buffer[PersonID]
    PCounter-=1

def Get_Features(PersonID):
    global Feature_Map_Buffer
    global PCounter
    return Feature_Map_Buffer[PersonID]

def Update_Features(PersonID, Feature, update_type='append'):
    global Feature_Map_Buffer
    global PCounter
    if(update_type == 'append'):
        #print('appending feature to PID: ',PersonID)
        prev=Get_Features(PersonID)
        #print("Previous tensor FM:",prev)
        #print("Previous tensor FM:", type(prev))
        #print("Feature Map:", type(Feature))
        #Feature_Map_Buffer[PersonID] = torch.cat((prev,Feature),0)
        prev.append(Feature[0])
        updated_features = prev
        Feature_Map_Buffer[PersonID] = updated_features
    else:
        #print('overwritting feature to PID: ',PersonID)
        Feature_Map_Buffer[PersonID] = Feature[0]
'''
def eval(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
'''
def l2_normalize(feature_vector):
    norm = np.linalg.norm(feature_vector)
    normalized_vector = feature_vector / norm
    return normalized_vector

def CheckREid(FeatureMap):
    global PCounter
    #print("Checking person reid")
    Reidentified = False
    PID = ''
    # Calculate distances between query and gallery embeddings
    distances = []
    #if not Feature_Map_Buffer:
    if (len((Feature_Map_Buffer.keys())) == 0):
        #print('look up is empty')
        PCounter += 1
        PID = str(PCounter)
        print("REid, PID", Reidentified, PID)
        return Reidentified, PID

    else:
        for key in Feature_Map_Buffer.keys():
            print("Checking with : ",str(key))
            PFM_Distances = []
            #print("PID: ",str(key))
            #print("Value: ",Feature_Map_Buffer[key])
            for feature in Feature_Map_Buffer[key]:
                #FeatureMap = F.normalize(FeatureMap, p=2, dim=1)
                #feature = F.normalize(feature, p=2, dim=1)
                #FeatureMap = l2_normalize(FeatureMap)
                #feature = l2_normalize(feature)
                #distances = np.linalg.norm(FeatureMap - feature)
                #print("type of the value in query , dim , QFeatureMap ",type(FeatureMap),FeatureMap.dim(),FeatureMap)
                #print("type of the value in key value pair , dim , Value feature ", type(feature),feature.dim(),feature)
                distances = compute_distance_matrix(torch.tensor(FeatureMap), torch.tensor(feature), metric=metric)
                distances = distances.numpy()
                PFM_Distances.append(distances[0])
                #print("distance between q fm and g fm:",distances[0])
     
            PFM_Distances.sort()
            #if PFM_Distances[0] < PReID_Thershold:
            print("Least Distance : ",PFM_Distances[0])
            #avg=sum(PFM_Distances)/len(PFM_Distances)
            #print("Avg Distance : ",avg)
            if (PFM_Distances[0] <= (PReID_Thershold + PReID_Tolorance)):
            #if (avg <= (PReID_Thershold + PReID_Tolorance)):
                Reidentified = True
                PID = str(key)
                break

        if not Reidentified:
            PCounter+=1
            PID=str(PCounter)

        print("REid, PID", Reidentified, PID)
        return Reidentified, PID

def TrackPerson(TempImg_Path):
    if os.path.exists(TempImg_Path):
        start = time.time()
        FeatureMap = Extract_Feature_Maps(TempImg_Path)
        end = time.time()
        print("Time taken to extract feature map: ", end-start)
        start = time.time()
        Reid , Pid = CheckREid(FeatureMap)
        end = time.time()
        print("Time taken to reidentify: ", end - start)
        time.sleep(5)
        os.remove(TempImg_Path)
        if Reid:
            print("Person reidentified!!!!")
            start = time.time()
            Update_Features(Pid, [FeatureMap.tolist()], 'append')
            end = time.time()
            print("Time taken to append feature map: ", end - start)
            #print("Feature map buff:", Feature_Map_Buffer)
            #print("Person ID:", Pid)
            return Pid
        else:
            print("New person!!!!")
            #Add_To_FBuffer(Pid, [FeatureMap])
            start = time.time()
            Add_To_FBuffer(Pid, [FeatureMap.tolist()])
            end = time.time()
            print("Time taken to add new feature map: ", end - start)
            #print("Feature map buff:", Feature_Map_Buffer)
            #print("Person ID:", Pid)
            return Pid