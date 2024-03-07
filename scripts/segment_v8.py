#!/usr/bin/python3.8
# license removed for brevity

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
import sys, datetime
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



global pub, box, video, timelog

# Initializing Threshold to detect the densest part of the smoke
THRESHOLD = 240




# --------------------------- OPTIONS -------------------------- #
VIEW_IMG = True
VIEW_SEGMENTATION = True
VIEW_MASK = False
VIEW_THRESHOLD_SEGMENT = False
RESIZE_IMSHOW_WINDOW = False
SAVE_SEGMENTATION = False
SAVE_FORMAT = False     # '.avi' or '.raw'
PRINT_TIME = False
PRINT_SEG_DATA = True
# --------------------------- OPTIONS -------------------------- #



# ------------------------ YOLO PARAMETERS --------------------- #
MAX_DELAY = 0.5        # [seconds] delay between last detection and current image after which to just drop images to catch up
CONF_THRES = 0.25      # previously 0.25  # confidence threshold
IOU_THRES = 0.6        # previously 0.7 # NMS IOU threshold
MAX_DET = 10           # maximum detections per image
IMGSZ = (160,128)      # previously [192,224], [352,448] # scaled image size to run inference on #inference size (height, width) 
DEVICE = 'cuda:0'      # DEVICE = 'cuda:0' or 'cpu'
RETINA_MASKS = False
HALF_PRECISION = True
SHOW_CONF = True
THRESHOLD = 240        # Initializing Threshold to detect the densest part of the smoke
# ------------------------ YOLO PARAMETERS --------------------- #



gps_t = 0

# create saving directory
username = os.getlogin( )
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
# YOLOv8 segmentation root directory
YOLOv8_SEG_ROOT = FILE.parents[1] / 'scripts/modules/yolov8-seg/yolo-V8'
if str(YOLOv8_SEG_ROOT) not in sys.path:
    sys.path.append(str(YOLOv8_SEG_ROOT ))  # add YOLOv8_SEG_ROOT to PATH
YOLOv8_SEG_ROOT  = Path(os.path.relpath(YOLOv8_SEG_ROOT, Path.cwd()))  # relative




def time_callback(gpstime):
    """ gps time callback """
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())



def imagecallback(img):
    """image callback function"""
    global pub, box, video, timelog
    global MODEL

    # converting image to numpy array
    t1 = time.time()
    img_numpy = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)
    # print(img_numpy.shape) # (480,640,3)

    if VIEW_IMG:
        img_view = img_numpy
        dim = IMGSZ
        if RESIZE_IMSHOW_WINDOW:
            img_view = cv2.resize(img_view, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('Img->Seg Node',img_view)
        cv2.waitKey(1)

    if rospy.Time.now() - img.header.stamp > rospy.Duration(MAX_DELAY):
        # print("Segmentation Node: dropping old image from segmentation\n")
        return
    else:
        results = MODEL.predict(img_numpy, conf=CONF_THRES, iou=IOU_THRES, 
                                imgsz=IMGSZ, half=HALF_PRECISION, device=DEVICE,
                                verbose=False, max_det=MAX_DET, 
                                retina_masks=RETINA_MASKS, show_conf=SHOW_CONF)

        if results[0].masks is not None:
            # resizing the original image to the size of mask
            resize_orig_img = cv2.resize(results[0].orig_img,
                                        (len(results[0].masks.data[0][0]),
                                        len(results[0].masks.data[0]))) 

            max_white_pixels, data_idx = 0, 0
            x_mean, y_mean = -1, -1

            for i in range(len(results[0].masks.data)):

                indices = find_element_indices(results[0].masks.data[0].cpu().numpy(), 1)
                white_x_mean, white_y_mean, white_pixel_indices = check_white_pixels(indices, resize_orig_img)

                # looking for the instance of segmentation which has
                # the max no. of segemented white pixels
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
                t2, t1 = 0, 0
            else:
                # selecting the instance which has the maximum number of segmented white pixels
                img_mask = results[0].masks.data[data_idx].cpu().numpy()
                img_mask = (img_mask * 255).astype("uint8")
                indices = find_element_indices(img_mask, 255)
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

                # centroid of smoke normalized to the size of the mask
                x_mean_norm = x_mean / img_mask.shape[0]
                y_mean_norm = y_mean / img_mask.shape[1]

                if VIEW_MASK:
                    img_mask = cv2.circle(img_mask, (y_mean, x_mean), 5, (0, 0, 255), -1)
                    dim = IMGSZ
                    if RESIZE_IMSHOW_WINDOW:
                        img_mask = cv2.resize(img_mask, dim, interpolation = cv2.INTER_AREA)
                    cv2.imshow('Segmentation Mask', img_mask)
                    cv2.waitKey(1)


                if VIEW_SEGMENTATION:
                    # result is a <class 'list'> in which the first element is 
                    # <class 'ultralytics.engine.results.Results'>
                    annotated_frame = results[0].plot()
                    # centroid of smoke normalized to the size of the original image
                    x_mean = int(x_mean_norm * annotated_frame.shape[0])
                    y_mean = int(y_mean_norm * annotated_frame.shape[1])
                    annotated_frame_view = cv2.circle(annotated_frame,
                                                 (y_mean, x_mean), 10, (255, 0, 0), -1)
                    dim = IMGSZ
                    if RESIZE_IMSHOW_WINDOW:
                        annotated_frame_view = cv2.resize(annotated_frame,
                                                        dim, interpolation = cv2.INTER_AREA)
                    cv2.imshow('Segmentation Window', annotated_frame_view)
                    cv2.waitKey(1)


                if SAVE_SEGMENTATION:
                    savenum=img.header.seq
                    annotated_frame = results[0].plot()
                    x_mean = int(x_mean_norm * annotated_frame.shape[0])
                    y_mean = int(y_mean_norm * annotated_frame.shape[1])
                    annotated_frame = cv2.circle(annotated_frame, (y_mean, x_mean), 10, (255, 0, 0), -1)
                    saving_stamp = (f"{tmp.month:02.0f}{tmp.day:02.0f}{tmp.year:04.0f}")

                    if SAVE_FORMAT =='.raw':
                        fid = open(savedir.joinpath('Segmentation-%06.0f.raw' % savenum),'wb')
                        fid.write(annotated_frame.flatten())
                        fid.close()
                    elif SAVE_FORMAT == '.avi':
                        video.write(annotated_frame)
                    else:
                        cv2.imwrite(str(savedir.joinpath(f'Segment-{saving_stamp}-{savenum:6.0f}.jpg')),annotated_frame)

                box.header.seq = img.header.seq
                box.header.stamp = img.header.stamp
                box.header.frame_id = ''
                box.source_img = img

                box.bbox.center.x = y_mean_norm 
                box.bbox.center.y = x_mean_norm
                box.bbox.center.theta = 0
                box.bbox.size_x = white_pixel_count
                pub.publish(box)

                t2 = time.time()
                if PRINT_TIME: 
                    print(f"Time taken from receiving to publishing: {(t2-t1)}")

        else:
            # No segmnetation data received
            box.bbox.center.x = -1
            box.bbox.center.y = -1
            box.bbox.center.theta = -1
            box.bbox.size_x = -1
            box.bbox.size_y = -1
            pub.publish(box)
            t2, t1 = 0, 0

        # adding to time stamp log, every frame
        timelog.write(f'{img.header.seq},{float(img.header.stamp.to_sec())},{gps_t},{box.bbox.center.x},{box.bbox.center.y},{box.bbox.center.theta},{box.bbox.size_x}\n')

        if PRINT_SEG_DATA:
            print(f'Segment center: ({box.bbox.center.x: .2f}, {box.bbox.center.y: .2f}) | Inference time: {(t2-t1): .4f} | Area of segment: {box.bbox.size_x}')



def init_segmentation_node():
    global pub, box, video, timelog
    global MODEL

    pub = rospy.Publisher('/segmentation_box', Detection2D, queue_size=1)
    box = Detection2D()

    print('Initializing YOLOv8 segmentation model')
    MODEL = YOLO(YOLOv8_SEG_ROOT / 'yolov8n-seg.pt')

    # initializing video file
    if SAVE_FORMAT =='.avi':
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        video = cv2.VideoWriter(str(savedir.joinpath('Detection'+SAVE_FORMAT)),
            fourcc=codec, fps=20, frameSize = (640,480)) # this size is specific to GoPro

    # initializing timelog
    timelog = open(savedir.joinpath('Metadata.csv'),'w')
    timelog.write('FrameID,Timestamp_Jetson,Timestamp_GPS,Centroid_x,Centroid_y,Width,Height\n')

    # initializing node
    rospy.init_node('segmentation_node', anonymous=False)


    if EXECUTION == 'SIMULATION':
        rospy.Subscriber('front_centre_cam', Image, imagecallback)
    if EXECUTION == 'DEPLOYMENT':
        rospy.Subscriber('/camera/image', Image, imagecallback)
        rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)

    rospy.spin()



def find_element_indices(arr, target_element=1):
    """
    returns the indices with pixel value of 1
    a pixel belonging to a segmented class is having a value of 1, rest 0
    """
    indices = []
    for row_index, row in enumerate(arr):
        for col_index, element in enumerate(row):
            if element == target_element:
                indices.append((row_index, col_index))

    return indices



def find_pixel_indices(arr, target_pixel = [0, 0, 0]):
    """
    returns the indices which are of a certain value
    """
    indices = []
    for row_index, row in enumerate(arr):
        for col_index, element in enumerate(row):
            if (element[0] == target_pixel[0] and
                element[1] == target_pixel[1] and
                element[2] == target_pixel[2]):
                indices.append((row_index, col_index))

    return indices



def check_white_pixels(mask_indices, img):
    """
    returns the centroid of the segmented region and 
    also the indices of the smoke segment above a predifined/dynamic threshold 
    (to detect the denser region of smoke)
    """

    white_pixel_indices = []
    for idx in mask_indices:
        if ((img[idx][0] > THRESHOLD) and
            (img[idx][1] > THRESHOLD) and
            (img[idx][0] > THRESHOLD)):
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

    if VIEW_THRESHOLD_SEGMENT:
        img = cv2.circle(img, (y_mean, x_mean), 2, (255, 0, 0), -1)
        dim = IMGSZ
        if RESIZE_IMSHOW_WINDOW:
            img_resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('Thresholded Segment', img_resize)
        cv2.waitKey(1)

    if len(white_pixel_indices) != 0:
        return x_mean, y_mean, white_pixel_indices
    else:
        return -1, -1, white_pixel_indices




if __name__ == '__main__':
    try:
        init_segmentation_node()
    except rospy.ROSInterruptException:
        pass
