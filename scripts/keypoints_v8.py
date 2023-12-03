#!/usr/bin/python3
# license removed for brevity

from operator import truediv
from re import sub
import rospy
from rospy.client import init_node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from sensor_msgs.msg import TimeReference
from std_msgs.msg import String
import numpy as np
import cv2
import os, re
import sys, datetime
import argparse
from pathlib import Path
import time
import torch
from scipy.stats import linregress

from ultralytics import YOLO

print(f"Torch setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")


#global publisher and boundingbox
global pub, box, video, timelog

#global initialized variables for segmentation model
global imgsz, model, device, names, max_det, max_delay
#global engine, half


#------------------------OPTIONS---------------------#
max_delay = 0.5 # [seconds] delay between last detection and current image after which to just drop images to catch up

conf_thres=0.25 # originally 0.4  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=100 # maximum detections per image
imgsz = (352,448) # previously [352,448] # scaled image size to run inference on #inference size (height, width) 
device='cpu' # device='cuda:0'
retina_masks=True

save_txt = False
save_img = False 
save_crop = False 
view_img = True
hide_labels=False,  # hide labels
hide_conf=False,  # hide confidences
VIEW_IMG = True
VIEW_MASK = False
VIEW_POINTS = True
SAVE_IMG = False
save_format = False #'.avi' or '.raw'
#-----------------------------------------------------#


gps_t = 0
# create saving directory
# username = os.getlogin( )
tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))
maindir = Path('./SavedData')
runs_today = list(maindir.glob('*%s*_keypoints' % stamp))
if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\d\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    new_run_num = max(runs_today)+1
else:
    new_run_num = 1
savedir = maindir.joinpath('%s_run%02d_keypoints' % (stamp,new_run_num))
os.makedirs(savedir)  


# YOLO paths and importing
FILE = Path(__file__).resolve()
YOLOv8_POSE_ROOT = FILE.parents[1] / 'scripts/modules/yolov8-pose/ultralytics'  # YOLOv8 root directory
if str(YOLOv8_POSE_ROOT) not in sys.path:
    sys.path.append(str(YOLOv8_POSE_ROOT))  # add YOLOv8_ROOT to PATH
# print(YOLOv8_ROOT)
YOLOv8_POSE_ROOT = Path(os.path.relpath(YOLOv8_POSE_ROOT, Path.cwd()))  # relative


# labeling text on image
BLACK = (265,265,265)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
font_color = BLACK
font_thickness = 2



def imagecallback(img):
    global pub,box,video,timelog
    global imgsz, model, device, names
    box = Detection2D()

    # converting image to numpy array
    img_numpy = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)

    if rospy.Time.now() - img.header.stamp > rospy.Duration(max_delay):
        print("KeypointNode: dropping old image from estimatimng keypoints\n")
        return
    else:
        results = model.predict(img_numpy, show=False, conf=conf_thres, imgsz=imgsz, iou=iou_thres, max_det=max_det, verbose=False)
        
        for result in results:
            keypoints = np.array(result.keypoints.data[0].cpu())
            #print(f'Keypoints: {keypoints[0]}, {int(keypoints[0][1])}, {int(keypoints[0][0])}, img: {img_numpy.shape}')
            if keypoints.all() is not None:
                try:
                    source_x, source_y = int(keypoints[0][1]), int(keypoints[0][0])

                    # wind_h, wind_w = 960, 540
                    # cv2.namedWindow('Smoke Keypoints', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # cv2.resizeWindow('Smoke Keypoints', wind_h, wind_w)
                    img_numpy = cv2.circle(img_numpy, (source_y, source_x), 10, (255, 0, 0), -1)
                    img_numpy = cv2.circle(img_numpy, (int(keypoints[1][0]), int(keypoints[1][1])), 10, (0, 255, 0), -1)
                    img_numpy = cv2.circle(img_numpy, (int(keypoints[2][0]), int(keypoints[2][1])), 10, (0, 0, 255), -1)
                    
                    scale_percent = 25 # percent of original size
                    width = int(img_numpy.shape[1] * scale_percent / 100)
                    height = int(img_numpy.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    # resize image
                    img_numpy_resize = cv2.resize(img_numpy, dim, interpolation = cv2.INTER_AREA)
                    
                    cv2.imshow('Smoke Keypoints', img_numpy_resize)
                    cv2.waitKey(1)
                    
                    #print(f'Source: {source_x}, {source_y}, img_shape: {img_numpy.shape}')
                    source_x, source_y = source_x/img_numpy.shape[0], source_y/img_numpy.shape[1]
                    #print(f'Normalized Source: {source_x}, {source_y}, img_shape: {img_numpy.shape}')
                    box.bbox.center.x = source_y 
                    box.bbox.center.y = source_x
                except:
                    box.bbox.center.x = -1
                    box.bbox.center.y = -1
            
            print('Publishing Box', end='\r')
            pub.publish(box)
        

def init_keypoints_node():
    global pub, box, video, timelog
    pub = rospy.Publisher('/keypoints', Detection2D, queue_size=1)
    box = Detection2D()

    global imgsz, model, device
    
    print('Initializing YOLOv8 pose model')
    model= YOLO(YOLOv8_POSE_ROOT / 'yolov8x-pose-best.pt')

    # initializing video file
    if save_format=='.avi':
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        video = cv2.VideoWriter(str(savedir.joinpath('Detection'+save_format)),
            fourcc=codec,
            fps=20,
            frameSize = (640,480)) # this size is specific to GoPro

    # initializing timelog
    timelog = open(savedir.joinpath('Metadata.csv'),'w')
    timelog.write('FrameID,Timestamp_Jetson,Timestamp_GPS,Centroid_x,Centroid_y,Width,Height\n')

    # initializing node
    rospy.init_node('smoke_keypoints', anonymous=False)
    rospy.Subscriber('front_centre_cam', Image, imagecallback)
    
    rospy.spin()




if __name__ == '__main__':
    try:
        init_keypoints_node()
    except rospy.ROSInterruptException:
        pass
