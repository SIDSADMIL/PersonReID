import cv2
import os
import yolo_crop



detectandcrop = True
input_folder = 'VideoData'
frame_folder = 'FrameData'
crop_folder = 'CroppedData'
fps = 10
i = 0


vdolist=os.listdir(input_folder)

for vdo in vdolist:
    currframecount = 0
    # Define video path
    video_path = input_folder+'/'+vdo
    # Define output folder path (create the folder if it doesn't exist)
    filename=os.path.splitext(vdo)
    output_folder = frame_folder+'/'+str(filename[0])
    os.makedirs(output_folder, exist_ok=True)
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    vfps = int(cap.get(cv2.CAP_PROP_FPS))
    desired_frame_interval = int(vfps / fps)
    # Check if video is opened successfully
    if not cap.isOpened():
        print("Error opening video!")
        exit()
    # Define frame counter and image name format
    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Check if the frame was grabbed successfully
        if ret:
            if frame_count % desired_frame_interval == 0:
                output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, frame)
                currframecount+=1
            frame_count += 1
        else:
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
    i += 1
    print(f"Itt No : {i} ,File name : {video_path} ,Extracted {currframecount} frames successfully!")


if detectandcrop:
    yolo_crop.detect_crop_person(frame_folder,crop_folder)

