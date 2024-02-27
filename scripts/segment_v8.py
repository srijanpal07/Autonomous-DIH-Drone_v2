#!/usr/bin/python3
# license removed for brevity

from operator import truediv
from re import sub
import rospy
from rospy.client import init_node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from sensor_msgs.msg import TimeReference
from std_msgs.msg import Float64, String
import numpy as np
import cv2
import os, re
# import PySpin
import sys, datetime
import argparse
from pathlib import Path
import time
import torch
from ultralytics.utils import ops

from ultralytics import YOLO

print(f"Torch setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

EXECUTION = rospy.get_param('EXECUTION', default='DEPLOYMENT') # 'SIMULATION' or 'DEPLOYMENT'
print(f'EXECUTION ==> {EXECUTION}')

if EXECUTION == 'SIMULATION':
    import airsim
    from airsim_ros_pkgs.msg import GimbalAngleEulerCmd, GPSYaw

#global publisher and boundingbox
global pub, box, video, timelog

#global initialized variables for segmentation model
global imgsz, model, device, names, max_det, max_delay
#global engine, half

global threshold
threshold = 240 # Initializing Threshold to detect the densest part of the white smoke
global sampling_time_passed
sampling_time_passed = 0 

#------------------------OPTIONS---------------------#
max_delay = 0.5 # [seconds] delay between last detection and current image after which to just drop images to catch up

conf_thres = 0.25 #previously 0.25  # confidence threshold
iou_thres = 0.6  # previously 0.7 # NMS IOU threshold
max_det = 100 # maximum detections per image
imgsz = (192,224) # previously [352,448] # scaled image size to run inference on #inference size (height, width) 
device='cpu' # device='cuda:0'
retina_masks=False

save_txt = False
save_img = True
save_crop = False 
hide_labels = False,  # hide labels
hide_conf = False,  # hide confidences
VIEW_IMG = True
VIEW_MASK = True
SAVE_IMG = False
save_format = False #'.avi' or '.raw'
#-----------------------------------------------------#


gps_t = 0
# create saving directory
# username = os.getlogin( )
tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))

if EXECUTION == 'SIMULATION':
    maindir = Path('./SavedData')
elif EXECUTION == 'DEPLOYMENT':
    username = os.getlogin()
    maindir = Path('/home/%s/1FeedbackControl' % username)
    
runs_today = list(maindir.glob('*%s*_segmentation' % stamp))
if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\d\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    new_run_num = max(runs_today)+1
else:
    new_run_num = 1
savedir = maindir.joinpath('%s_run%02d_segmentation' % (stamp,new_run_num))
os.makedirs(savedir)  


# YOLO paths and importing
FILE = Path(__file__).resolve()
YOLOv5_ROOT = FILE.parents[1] / 'scripts/modules/yolov8-seg/yolo-V8'  # YOLOv5 root directory
if str(YOLOv5_ROOT) not in sys.path:
    sys.path.append(str(YOLOv5_ROOT))  # add YOLOv5_ROOT to PATH
# print(YOLOv5_ROOT)
YOLOv5_ROOT = Path(os.path.relpath(YOLOv5_ROOT, Path.cwd()))  # relative


# labeling text on image
BLACK = (265,265,265)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
font_color = BLACK
font_thickness = 2




def time_info_callback(data):
    global threshold, sampling_time_passed

    #rospy.loginfo(f"Received timestamp: {data.data} seconds") 
    if int(data.data) - sampling_time_passed > 1:
        sampling_time_passed = sampling_time_passed + 1
        if sampling_time_passed % 2 == 0:
            threshold = threshold - 1
    print(f"Received timestamp: {sampling_time_passed} secs ------- threshold reduced to : threshold: {threshold}", end='\r')


def time_callback(gpstime):
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())
    
    

def imagecallback(img):
    global pub,box,video,timelog
    global imgsz, model, device, names
    box = Detection2D()

    # converting image to numpy array
    img_numpy = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)
    print('Got Numpy Image!')
    
    if VIEW_IMG:
        result_ = img_numpy
        scale_percent = 25 # percent of original size
        width = int(result_.shape[1] * scale_percent / 100)
        height = int(result_.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized_ = cv2.resize(result_, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('Segmentation',resized_)
        cv2.waitKey(1)  # 1 millisecond

    if rospy.Time.now() - img.header.stamp > rospy.Duration(max_delay):
        #print("DetectionNode: dropping old image from detection\n")
        return
    else:
        #results = model.predict(img_numpy, show=True, conf=conf_thres, imgsz=imgsz, iou=iou_thres, max_det=max_det, verbose=False, show_conf=False, retina_masks=False)
        results = model.predict(img_numpy, imgsz=imgsz, verbose=False, show_conf=True)
        #results = non_max_suppression(results, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False, max_det=max_det, max_wh=1000)

        if results[0].masks is not None:
            # resizing the original image to the size of mask
            resize_orig_img = cv2.resize(results[0].orig_img, (len(results[0].masks.data[0][0]), len(results[0].masks.data[0]))) 

            max_white_pixels, data_idx = 0, 0
            x_mean, y_mean = -1, -1
            for i in range(len(results[0].masks.data)):
                #results = non_max_suppression(results[0].masks.data[0], conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False, max_det=max_det, max_wh=1000)

                indices = find_element_indices(results[0].masks.data[0].cpu().numpy(), 1) # a pixel belonging to a segmented class is having a value of 1, rest 0
                #indices = find_element_indices(results[0].masks.data[0].numpy(), 1)

                # putting a threshold to the segmented region to detect the denser part of the smoke and finding the centroid of the denser region
                white_x_mean, white_y_mean, white_pixel_indices = check_white_pixels(indices, resize_orig_img)
                
                # looking for the instance of segmentation which has the maximum number of segemented white pixels
                white_pixel_count = len(white_pixel_indices)
                if max_white_pixels <= white_pixel_count:
                    max_white_pixels = white_pixel_count
                    data_idx = i
                    x_mean, y_mean = white_x_mean, white_y_mean
            
            if x_mean == -1 and y_mean == -1:
                # No segmnetation data received
                box.bbox.center.x = -1
                box.bbox.center.y = -1
                box.bbox.center.theta = -1
                box.bbox.size_x = -1
                box.bbox.size_y = -1
                pub.publish(box)
            else:
                # selecting the instance which has the maximum number of segemented white pixels
                img_mask = results[0].masks.data[data_idx].cpu().numpy()
                img_mask = (img_mask * 255).astype("uint8")
                indices = find_element_indices(img_mask, 255)
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
  
                # centroid of smoke normalized to the size of the mask
                x_mean_norm = x_mean / img_mask.shape[0]
                y_mean_norm = y_mean / img_mask.shape[1]

                if VIEW_MASK:
                    img_mask = cv2.circle(img_mask, (y_mean, x_mean), 5, (0, 0, 255), -1)
                    cv2.imshow('Smoke Mask', img_mask)
                    cv2.waitKey(1)
        
                annotated_frame = results[0].plot() # result is a <class 'list'> in which the first element is<class 'ultralytics.engine.results.Results'>
                # centroid of smoke normalized to the size of the original image
                x_mean = int(x_mean_norm * annotated_frame.shape[0])
                y_mean = int(y_mean_norm * annotated_frame.shape[1])
                annotated_frame = cv2.circle(annotated_frame, (y_mean, x_mean), 10, (255, 0, 0), -1)
                
                if VIEW_IMG:
                    annotated_frame = results[0].plot() # result is a <class 'list'> in which the first element is<class 'ultralytics.engine.results.Results'>
                    # centroid of smoke normalized to the size of the original image
                    x_mean = int(x_mean_norm * annotated_frame.shape[0])
                    y_mean = int(y_mean_norm * annotated_frame.shape[1])
                    annotated_frame = cv2.circle(annotated_frame, (y_mean, x_mean), 20, (255, 0, 0), -1)

                    #wind_h, wind_w = 960, 540 # (192,224)
                    scale_percent = 25 # percent of original size
                    width = int(annotated_frame.shape[1] * scale_percent / 100)
                    height = int(annotated_frame.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    annotated_frame_resize = cv2.resize(annotated_frame, dim, interpolation = cv2.INTER_AREA)
                    # cv2.namedWindow('Smoke Segmentation', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # cv2.resizeWindow('Smoke Segmentation', wind_h, wind_w)
                    cv2.imshow('Smoke Segmentation', annotated_frame_resize)
                    cv2.waitKey(1)

                box.header.seq = img.header.seq
                box.header.stamp = img.header.stamp
                box.header.frame_id = ''
                box.source_img = img

                box.bbox.center.x = y_mean_norm 
                box.bbox.center.y = x_mean_norm
                box.bbox.center.theta = 0
                box.bbox.size_x = white_pixel_count
                pub.publish(box)
                
                #print(f'No fo white pixels: {white_pixel_count}', end='\r')
                text_to_image = 'processed'
                img_numpy = cv2.putText(img_numpy,text_to_image,(10,30),font, font_size, font_color, font_thickness, cv2.LINE_AA)

                # adding to time stamp log, every frame
                timelog.write('%d,%f,%f,%f,%f, %f, %f\n' % (img.header.seq,
                                                        float(img.header.stamp.to_sec()),
                                                        gps_t,
                                                        box.bbox.center.x,
                                                        box.bbox.center.y, 
                                                        box.bbox.center.theta,
                                                        box.bbox.size_x
                                                        ))
            
        else:
            # No segmnetation data received
            box.bbox.center.x = -1
            box.bbox.center.y = -1
            box.bbox.center.theta = -1
            box.bbox.size_x = -1
            box.bbox.size_y = -1
            pub.publish(box)

        # viewing/saving images
        savenum=img.header.seq

        if SAVE_IMG:
            if save_format=='.raw':
                fid = open(savedir.joinpath('Segmentation-%06.0f.raw' % savenum),'wb')
                fid.write(img_numpy.flatten())
                fid.close()
            elif save_format == '.avi':
                video.write(img_numpy)
            else:
                cv2.imwrite(str(savedir.joinpath('Segmentation-%06.0f.jpg' % savenum)),img_numpy)
        
        

def init_detection_node():
    global pub, box, video, timelog
    pub = rospy.Publisher('/segmentation_box', Detection2D, queue_size=1)
    box = Detection2D()

    global imgsz, model, device
    
    print('Initializing YOLOv8 segmentation model')
    model= YOLO(YOLOv5_ROOT / 'yolov8m-seg.pt') # model= YOLO(YOLOv5_ROOT / 'yolov8-best.pt')

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
    rospy.init_node('segmentation_node', anonymous=False)
    rospy.Subscriber('sampling_time_info_topic', Float64, time_info_callback)
    if EXECUTION == 'SIMULATION':
        rospy.Subscriber('front_centre_cam', Image, imagecallback)
    if EXECUTION == 'DEPLOYMENT':
        print('Subscribed')
        rospy.Subscriber('/camera/image', Image, imagecallback)
        rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)
    
    rospy.spin()



def find_element_indices(arr, target_element=1):
    indices = []
    for row_index, row in enumerate(arr):
        for col_index, element in enumerate(row):
            if element == target_element:
                indices.append((row_index, col_index))

    return indices



def find_pixel_indices(arr, target_pixel = [0, 0, 0]):
    indices = []
    for row_index, row in enumerate(arr):
        for col_index, element in enumerate(row):
            if (element[0] == target_pixel[0] and element[1] == target_pixel[1] and element[2] == target_pixel[2]):
                indices.append((row_index, col_index))

    return indices



def check_white_pixels(mask_indices, img):
    global threshold
    white_pixel_indices = []
    for idx in mask_indices:
        if (img[idx][0] > threshold) and (img[idx][1] > threshold) and (img[idx][0] > threshold):
            img[idx][0], img[idx][1], img[idx][0] = 0, 0, 0
            white_pixel_indices.append(idx)

    x_cord_sum, y_cord_sum = 0, 0
    num_white_pixels = len(white_pixel_indices)
    for i in range(num_white_pixels):
        x_cord_sum = x_cord_sum + white_pixel_indices[i][0]
        y_cord_sum = y_cord_sum + white_pixel_indices[i][1]
    
    if num_white_pixels != 0:
        x_mean = int(x_cord_sum / num_white_pixels)
        y_mean = int(y_cord_sum / num_white_pixels)
    else:
        x_mean = -1
        y_mean = -1

    #print(f'Len of mask_indices: {len(mask_indices)}, Len of white_pixel_indices: {len(white_pixel_indices)}', end='\r')
    if VIEW_IMG:
        img = cv2.circle(img, (y_mean, x_mean), 2, (255, 0, 0), -1)
        width = int(img.shape[1]*2)
        height = int(img.shape[0]*2)
        dim = (width, height)

        # resize image
        img_resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('Thresholding Segmentation', img_resize)
        cv2.waitKey(1)

    if len(white_pixel_indices) != 0:
        return x_mean, y_mean, white_pixel_indices
    else:
        return -1, -1, white_pixel_indices




if __name__ == '__main__':
    try:
        init_detection_node()
    except rospy.ROSInterruptException:
        pass
