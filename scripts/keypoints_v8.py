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
from std_msgs.msg import Float64
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


#------------------------EXECUTION SETUP------------------------#

EXECUTION = rospy.get_param('EXECUTION', default='DEPLOYMENT') # 'SIMULATION' or 'DEPLOYMENT'
if EXECUTION == 'SIMULATION':
    import airsim
    from airsim_ros_pkgs.msg import GimbalAngleEulerCmd, GPSYaw

#------------------------EXECUTION SETUP------------------------#


#global publisher and boundingbox
global pub, box, video, timelog
global smoketrack_status
smoketrack_status = 'Initializing'

#global initialized variables for segmentation model
global imgsz, model, device, names, max_det, max_delay


#------------------------OPTIONS---------------------#
max_delay = 0.5       # [seconds] delay between last detection and current image after which to just drop images to catch up
conf_thres = 0.25     # originally 0.4  # confidence threshold
iou_thres = 0.45      # NMS IOU threshold
max_det = 100         # maximum detections per image
imgsz = (352,448)     # previously [352,448] # scaled image size to run inference on #inference size (height, width) 
device = 'cuda:0'     # device='cuda:0' or device='cpu'

VIEW_IMG = False
VIEW_KEYPOINTS = False
SAVE_IMG = False
SAVE_KEYPOINTS = True
save_format = '.jpg' #'.avi' or '.raw'
#-----------------------------------------------------#


gps_t = 0


#------------------------SAVING DIRECTORY SETUP------------------------#

tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))

if EXECUTION == 'SIMULATION':
    maindir = Path('./SavedData')
elif EXECUTION == 'DEPLOYMENT':
    username = os.getlogin()
    maindir = Path('/home/%s/1FeedbackControl' % username)
    
runs_today = list(maindir.glob('%s/run*' % stamp))

if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\d\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    run_num = max(runs_today)
else:
    run_num = 1

savedir = maindir.joinpath('%s/run%02d/keypoints' % (stamp, run_num))
os.makedirs(savedir)  

#------------------------SAVING DIRECTORY SETUP------------------------#


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



def smoketrack_status_callback(status):
    global smoketrack_status
    print(f"Inside Callback : {str(status.data)}")
    smoketrack_status = str(status.data)



def time_callback(gpstime):
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())



def imagecallback(img):
    global pub,box,video,timelog
    global smoketrack_status
    global imgsz, model, device, names

    box = Detection2D()

    # converting image to numpy array
    img_numpy = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)
    
    if VIEW_IMG:
        scale_percent = 25 # percent of original size
        width = int(img_numpy.shape[1] * scale_percent / 100)
        height = int(img_numpy.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_numpy_resize = cv2.resize(img_numpy, dim, interpolation = cv2.INTER_AREA)  # resize image
        cv2.imshow('Cam Img to Keypt Node', img_numpy_resize)
        cv2.waitKey(1)

    if rospy.Time.now() - img.header.stamp > rospy.Duration(max_delay):
        print("KeypointNode: dropping old image from estimatimng keypoints\n")
        return
    elif smoketrack_status == 'Using Keypoints':
        results = model.predict(img_numpy, show=False, conf=conf_thres, imgsz=imgsz, iou=iou_thres, max_det=max_det, verbose=False)
        
        for result in results:
            keypoints = np.array(result.keypoints.data[0].cpu())
            
            if keypoints.all() is not None:
                try:
                    source_x, source_y = int(keypoints[0][1]), int(keypoints[0][0])
                    source_x, source_y = source_x/img_numpy.shape[0], source_y/img_numpy.shape[1]
                    box.bbox.center.x = source_y 
                    box.bbox.center.y = source_x
                    
                    if VIEW_KEYPOINTS:
                        img_numpy = cv2.circle(img_numpy, (source_y, source_x), 10, (255, 0, 0), -1)
                        img_numpy = cv2.circle(img_numpy, (int(keypoints[1][0]), int(keypoints[1][1])), 10, (0, 255, 0), -1)
                        img_numpy = cv2.circle(img_numpy, (int(keypoints[2][0]), int(keypoints[2][1])), 10, (0, 0, 255), -1)
                        scale_percent = 25 # percent of original size
                        width = int(img_numpy.shape[1] * scale_percent / 100)
                        height = int(img_numpy.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img_numpy_resize = cv2.resize(img_numpy, dim, interpolation = cv2.INTER_AREA)  # resize image
                        cv2.imshow('Smoke Keypoints', img_numpy_resize)
                        cv2.waitKey(1)

                    if SAVE_KEYPOINTS:
                        if save_format=='.raw':
                                fid = open(savedir.joinpath('Keypoints-%06.0f.raw' % savenum),'wb')
                                fid.write(img_numpy.flatten())
                                fid.close()
                        elif save_format == '.avi': video.write(img_numpy)
                        else: cv2.imwrite(str(savedir.joinpath('Keypoints-%06.0f.jpg' % savenum)),img_numpy_resize)
                except:
                    img_numpy_resize = img_numpy
                    box.bbox.center.x = -1
                    box.bbox.center.y = -1
                    print("No Keypoints detected!")
            
            
            # viewing/saving images
            savenum=img.header.seq
            
            if SAVE_IMG:
                if save_format=='.raw':
                        fid = open(savedir.joinpath('Keypoints-%06.0f.raw' % savenum),'wb')
                        fid.write(img_numpy.flatten())
                        fid.close()
                elif save_format == '.avi': video.write(img_numpy)
                else: cv2.imwrite(str(savedir.joinpath('Keypoints-img-%06.0f.jpg' % savenum)),img_numpy_resize)
        
            print('Publishing Box', end='\r')
            pub.publish(box)
    else:
        print("Not Running Keypoints!!")


def init_keypoints_node():
    global pub, box, video, timelog
    global imgsz, model, device

    # Initiliazing Publisher
    pub = rospy.Publisher('/keypoints', Detection2D, queue_size=1)

    print('Initializing YOLOv8 Keypoints model')
    model= YOLO(YOLOv8_POSE_ROOT / 'keypoints-best.pt') #yolov8x-pose-best.pt

    # Initializing video file
    if save_format=='.avi':
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        video = cv2.VideoWriter(str(savedir.joinpath('Keypoints'+save_format)),
            fourcc=codec,
            fps=20,
            frameSize = (640,480)) # this size is specific to GoPro

    # Initializing timelog
    timelog = open(savedir.joinpath('Metadata.csv'),'w')
    timelog.write('FrameID,Timestamp_Jetson,Timestamp_GPS,Centroid_x,Centroid_y,Width,Height\n')

    # Initializing node
    rospy.init_node('smoke_keypoints', anonymous=False)
    rospy.Subscriber('/smoketrack_status', String, smoketrack_status_callback)

    # Subscribing to the camera image topic
    if EXECUTION == 'SIMULATION':
        rospy.Subscriber('front_centre_cam', Image, imagecallback)
    if EXECUTION == 'DEPLOYMENT':
        rospy.Subscriber('/camera/image', Image, imagecallback)
        rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)
    
    rospy.spin()



if __name__ == '__main__':
    try:
        init_keypoints_node()
    except rospy.ROSInterruptException:
        pass
