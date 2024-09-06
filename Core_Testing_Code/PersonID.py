import os
import time
import numpy as np
import cv2
import Support


global filename_OP
filename_OP = 'Output_Vdo.avi'

if __name__ == "__main__":
    #person_detection_model_path = 'frozen_inference_graph.pb'
    tempfolder = 'TempImage'
    os.makedirs(tempfolder, exist_ok=True)
    tempimg_path = 'TempImage/TempImg.jpg'
    threshold = 0.7
    Video_Capture = cv2.VideoCapture('3.mp4') #'input.mp4'
    fps = int(Video_Capture.get(cv2.CAP_PROP_FPS))
    width = int(Video_Capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(Video_Capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #width = 1200  # float `width`
    #height = 800
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #(*'XVID')
    writer = cv2.VideoWriter(filename_OP, fourcc, fps, (width, height))
    print('starting')
    Support.Flush_FBuffer()
    Fcount = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    box_color = (0, 0, 255)
    box_thickness = 2
    while True:
        if Video_Capture.isOpened():
            #try :
            Success_reading, Frame = Video_Capture.read()
            if Success_reading:
                print('Valid frame................')
                Fcount += 1
                Frame = cv2.resize(Frame, (width, height))
                start = time.time()
                Bboxes, Scores = Support.detect_persons_yolo(Frame)
                #Bboxes, Scores = Support.detect_persons(Frame)
                end = time.time()
                print("Time taken to detect and get bbox from a frame: ", end-start)
                print("No of persons detected in the frame: ", len(Bboxes))
                i = 0
                for box in Bboxes:
                    print('valid detection.............')
                    y_min, x_min, y_max, x_max = box
                    Support.crop_save_person(Frame, box, tempimg_path)
                    cv2.rectangle(Frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, box_thickness)
                    
                    Label = Support.TrackPerson(tempimg_path)
                 
                    #cv2.putText(Frame, "PID "+Label+" : Acc "+str(format(Scores[i],".1f")), (int(x_min), int(y_max) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
                    #cv2.putText(Frame, Label, (int(x_min), int(y_min) + 15), font, font_scale,text_color , thickness) 
                    #cv2.putText(Frame, str(format(Scores[i],".1f")), (int(x_min), int(y_max) - 5), font, font_scale, text_color, thickness)
                    text_size, _ = cv2.getTextSize(Label, font, font_scale, thickness)
                    text_width, text_height = text_size
                    background = np.zeros((text_height + 15, text_width + 10, 3), dtype=np.uint8)
                    #cv2.putText(Frame, Label, (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)                    
                    # Overlay the background on the frame (reversed order)
                    #Frame[int(y_min):int(y_min)+text_height+10, int(x_min):int(x_min)+text_width+10] = background
                    
                    #below line will add black background to the pid
                    Frame[int(y_min):int(y_min)+text_height+15, int(x_min):int(x_min)+text_width+10] = background
                    
                    # Draw the text on the frame (now on top of the background)
                    cv2.putText(Frame, Label, (int(x_min), int(y_min) + 15), font, font_scale,text_color , thickness)
                    '''
                    # Create a background rectangle for the ID
                      
                    cv2.rectangle(Frame, (int(x_min), int(y_min)+15, int(x_max), int(y_min)+15), bg_color, -1)  # Fill the rectangle
                 
                    # Draw the ID text with a contrasting color
                    text_color =  (255, 255, 255) 
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(Frame, f'{Label}', (int(x_min), int(y_min) +15), font, 0.5, text_color, 2)
                    '''
                    i+=1
                    
                        
                cv2.imshow('Person ReID Demo', Frame)
                writer.write(Frame)
                
                key = cv2.waitKey(10)
                if key & 0xFF == ord('q'):
                    print('stopping............')
                    break
                
            else:
                print("Done ........................")
                break
                
            #key = cv2.waitKey(10)
            #if key & 0xFF == ord('q'):
            #    print('stopping............')
            #   break
                
    Video_Capture.release()
    cv2.destroyAllWindows()
