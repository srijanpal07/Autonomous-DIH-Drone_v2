#!/usr/local/bin/python3.10 
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
# import PySpin
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
VIEW_IMG=True
VIEW_MASK=False
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



def imagecallback(img):
    global pub,box,video,timelog
    global imgsz, model, device, names
    box = Detection2D()

    # converting image to numpy array
    img_numpy = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)

    if rospy.Time.now() - img.header.stamp > rospy.Duration(max_delay):
        #print("DetectionNode: dropping old image from detection\n")
        return
    else:
        results = model.predict(img_numpy, conf=conf_thres, imgsz=imgsz, iou=iou_thres, max_det=max_det, verbose=False)

        if results[0].masks != None:

            resize_orig_img = cv2.resize(results[0].orig_img, (len(results[0].masks.data[0][0]), len(results[0].masks.data[0]))) # resizing the original image to the size of mask 
            #cv2.imshow('Original Image', resize_orig_img)
            #cv2.waitKey(1)
            #print(f'Resize Image Shape: {resize_orig_img.shape}')
            #print(f'Resize Image: {resize_orig_img}')
            #blk  = np.zeros(img_numpy.shape)

            max_white_pixels, data_idx = 0, 0
            x_mean, y_mean = -1, -1
            for i in range(len(results[0].masks.data)):
                indices = find_element_indices(results[0].masks.data[0].cpu().numpy(), 1) # pixels belonging to a segmented image is having corresponding value of 1
                white_x_mean, white_y_mean, white_pixel_indices = check_white_pixels(indices, resize_orig_img)
                white_pixel_count = len(white_pixel_indices)
                if max_white_pixels <= white_pixel_count:
                    max_white_pixels = white_pixel_count
                    data_idx = i
                    x_mean, y_mean = white_x_mean, white_y_mean
            
            if x_mean == -1 and y_mean == -1:
                box.bbox.center.x = -1
                box.bbox.center.y = -1
                box.bbox.center.theta = -1
                box.bbox.size_x = -1
                box.bbox.size_y = -1
                pub.publish(box)
            else:
                #print(f'Data List length: ({len(results[0].masks.data[0])}, {len(results[0].masks.data[0][0])})')
                #print(f'Masked Pixels: {mask_count}')
                #white_x_mean, white_y_mean, white_pixel_indices = check_white_pixels(indices, resize_orig_img)
                #print(f'Data List : {find_element_indices(results[0].masks.data[0].cpu().numpy(), 1)}')
                #print(f'Max data Length: {max_mask}')
                #print(f'Data idx: {data_idx}')
                #print(len(indices))

                img_mask = results[0].masks.data[data_idx].cpu().numpy()
                img_mask = (img_mask * 255).astype("uint8")
                indices = find_element_indices(img_mask, 255)
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
                '''
                x_cord_sum, y_cord_sum = 0, 0
                len_indices = len(indices)
                for i in range(len_indices):
                    x_cord_sum = x_cord_sum + indices[i][0]
                    y_cord_sum = y_cord_sum + indices[i][1]
                x_mean = int(x_cord_sum / len_indices)
                y_mean = int(y_cord_sum / len_indices)
                '''
                x_mean_norm = x_mean / img_mask.shape[0]
                y_mean_norm = y_mean / img_mask.shape[1]
                #print(img.shape)
                #print(x_mean, y_mean)
                #print(results[0].masks.data[0])
                #print(indices)
                #python_indices  = [index for (index, item) in enumerate(programming_languages) if item == "Python"]
                if VIEW_MASK:
                    img_mask = cv2.circle(img_mask, (y_mean, x_mean), 5, (0, 0, 255), -1)
                    cv2.imshow('Mask', img_mask)
                    cv2.waitKey(1)

                #print(img_numpy.shape)
                #cor_x = (results[0].masks.xy[0][:,0] * (img_numpy.shape[1])).astype("int")
                #cor_x = (results[0].masks.xy[0][:,0]).astype("int")
                #cor_y = (results[0].masks.xy[0][:,1]).astype("int")
                #print(cor_x)
                #blk[cor_y,cor_x] = 255

                #wind_h, wind_w = 960, 540
                #cv2.namedWindow('Outline', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                #cv2.resizeWindow('Outline', wind_h, wind_w)
                #cv2.imshow('Outline', blk)
                #cv2.waitKey(1)
        
                annotated_frame = results[0].plot() # result is a <class 'list'> in which the first element is<class 'ultralytics.engine.results.Results'>
                #print(f'Annotataed array: {annotated_frame.shape}')
                #print(f'Index Length: {len(find_pixel_indices(img_mask, [0, 0, 0]))}')
                x_mean = int(x_mean_norm * annotated_frame.shape[0])
                y_mean = int(y_mean_norm * annotated_frame.shape[1])
                annotated_frame = cv2.circle(annotated_frame, (y_mean, x_mean), 10, (255, 0, 0), -1)
                
                if VIEW_IMG:
                    wind_h, wind_w = 960, 540
                    cv2.namedWindow('Segmentation V8', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow('Segmentation V8', wind_h, wind_w)
                    cv2.imshow('Segmentation V8', annotated_frame)
                    cv2.waitKey(1)

                box.header.seq = img.header.seq
                box.header.stamp = img.header.stamp
                box.header.frame_id = ''
                box.source_img = img

                box.bbox.center.x = y_mean_norm #object[0].bounding_box[0]
                box.bbox.center.y = x_mean_norm #object[0].bounding_box[1]
                box.bbox.center.theta = 0
                #box.bbox.size_x = object[0].bounding_box[2]
                #box.bbox.size_y = object[0].bounding_box[3]
                #print(f'box.bbox.center.x: {box.bbox.center.x} | box.bbox.center.y: {box.bbox.center.y}')
                pub.publish(box)

                text_to_image = 'processed'
                # print('Time after running detection')
                # print('Image %d' % box.source_img.header.seq)
                # print(box.source_img.header.stamp)
                
                # end = time.time()
                # print("finished callback for image", img.header.seq,"in",end-start, "seconds \n")
                img_numpy = cv2.putText(img_numpy,text_to_image,(10,30),font, font_size, font_color, font_thickness, cv2.LINE_AA)

                # adding to time stamp log, every frame
                timelog.write('%d,%f,%f,%f,%f\n' % (img.header.seq,
                                                        float(img.header.stamp.to_sec()),
                                                        gps_t,
                                                        box.bbox.center.x,
                                                        box.bbox.center.y
                                                        ))
            
        else:
            # print("everything -1")
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
                fid = open(savedir.joinpath('Detection-%06.0f.raw' % savenum),'wb')
                fid.write(img_numpy.flatten())
                fid.close()
            elif save_format == '.avi':
                video.write(img_numpy)
            else:
                cv2.imwrite(str(savedir.joinpath('Detection-%06.0f.jpg' % savenum),img_numpy))
        
        

def init_detection_node():
    global pub, box, video, timelog
    pub = rospy.Publisher('/segmentation_box', Detection2D, queue_size=1)
    box = Detection2D()

    global imgsz, model, device
    
    print('Initializing YOLOv8 segmentation model')
    model= YOLO(YOLOv5_ROOT / 'yolov8-best.pt')

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
    rospy.init_node('segment_smoke', anonymous=False)
    rospy.Subscriber('front_centre_cam', Image, imagecallback)
    
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
    threshold = 190 # white smoke
    #cv2.imshow('Original Image', img)
    #cv2.waitKey(1)
    #print(f'Indices:{mask_indices}')
    white_pixel_indices = []
    for idx in mask_indices:
        if (img[idx][0] > threshold) and (img[idx][1] > threshold) and (img[idx][0] > threshold):
            img[idx][0], img[idx][1], img[idx][0] = 0, 0, 0
            white_pixel_indices.append(idx)

    x_cord_sum, y_cord_sum = 0, 0
    x, y= [], []
    num_white_pixels = len(white_pixel_indices)
    for i in range(num_white_pixels):
        x_cord_sum = x_cord_sum + white_pixel_indices[i][0]
        x.append(white_pixel_indices[i][0])
        y_cord_sum = y_cord_sum + white_pixel_indices[i][1]
        y.append(white_pixel_indices[i][1])


    if num_white_pixels != 0:
        x_mean = int(x_cord_sum / num_white_pixels)
        y_mean = int(y_cord_sum / num_white_pixels)
        x_arr, y_arr = np.array(x), np.array(y)
        slope, intercept, r_value, p_value, std_err = linregress(x_arr, y_arr)
        theta = np.degrees(np.arctan(slope))
        print(f'Len of mask_indices: {len(mask_indices)}, Len of white_pixel_indices: {len(white_pixel_indices)}, linear regression: {theta}, {intercept}, {r_value}, {p_value}, {std_err}', end='\r')
        '''
        start_x, start_y= int(y_mean - 100 * np.cos(theta)), int(x_mean - 100 * np.sin(theta))
        end_x, end_y = int(y_mean + 100 * np.cos(theta)), int(x_mean + 100 * np.sin(theta))
        if start_x > end_x and start_y > end_y:
            new_start_x, new_start_y = end_x, end_y
            end_x, end_y = start_x, start_y
            start_x, start_y = new_start_x, new_start_y

        img = cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
        '''
        img = cv2.circle(img, (y_mean, x_mean), 3, (255, 0, 0), -1)
    else:
        x_mean = -1
        y_mean = -1

    

    cv2.imshow('Processed Image', img)
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