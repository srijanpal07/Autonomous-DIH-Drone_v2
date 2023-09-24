#!/usr/bin/python3 
# license removed for brevity

from operator import truediv
from re import sub
import rospy
from rospy.client import init_node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from sensor_msgs.msg import TimeReference
import numpy as np
import cv2
#from cv2 import cvShowImage
import os, re
# import PySpin
import sys, datetime
import argparse
from pathlib import Path
import time
import torch
import imageio
# import torch_tensorrt
print(f"Torch setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

#------------------------OPTIONS---------------------#
target_name = 'smoke' # options: smoke,car,person
engine = False # using tensorrt
half = False
max_delay = 0.5 # [seconds] delay between last detectiona nd current image after which to just drop images to catch up
conf_thres=0.4  # confidence threshold
iou_thres=0.45  # NMS IOU threshold

VIEW_IMG=False
SAVE_IMG = True # Originally False
save_format = '.jpg' # originally'.avi'
#-----------------------------------------------------#

gps_t = 0
# create saving directory
# username = os.getlogin( )
tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))
maindir = Path('/media/swarm1/gaia1/Data/autonomousdihdrone_v2')
#maindir = Path('./SavedData')
runs_today = list(maindir.glob('*%s*_stream' % stamp))
if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\d\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    new_run_num = max(runs_today)+1
else:
    new_run_num = 1
savedir = maindir.joinpath('%s_run%02d_stream' % (stamp,new_run_num))
os.makedirs(savedir)  

'''
# YOLO paths and importing
FILE = Path(__file__).resolve()
YOLOv5_ROOT = FILE.parents[1] / 'scripts/modules/yolov5'  # YOLOv5 root directory
if str(YOLOv5_ROOT) not in sys.path:
    sys.path.append(str(YOLOv5_ROOT))  # add YOLOv5_ROOT to PATH
# print(YOLOv5_ROOT)
YOLOv5_ROOT = Path(os.path.relpath(YOLOv5_ROOT, Path.cwd()))  # relative
#from models.experimental import attempt_load
from models.common import DetectMultiBackend
# from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.augmentations import letterbox


#global publisher and boundingbox
global pub,box, video, timelog
#global initialized variables for detection model
global imgsz, model, device, names

# labeling text on image
BLACK = (265,265,265)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_color = BLACK
font_thickness = 1
'''

def imagecallback(img):
    # print("imagecallback()")
    #global pub,box,video,timelog
    global timelog, video
    #global imgsz, model, device, names
    box = Detection2D()
    # print(img.header.stamp)
    # print('Time before running detection')
    # print('Image %d' % img.header.seq)
    # print(img.header.stamp)

    # time_stamp = time.time()

    # converting image to numpy array
    img_numpy = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)

    if rospy.Time.now() - img.header.stamp > rospy.Duration(max_delay):
        print("DetectionNode: dropping old image from detection", end='\r')
        # text_to_image = 'skipped'
        return
    else:
        # print('DetectionNode: Running detection inference')
        t1 = time.time()
        #object,img_numpy = detection(img_numpy,imgsz,model,device,names,savenum=img.header.seq)
        # print('Detection function took',1e3*(time.time()-t1))
        
        t1 = time.time()
        # print(img.header)
        # print('Printing time stamps at exchange in detection')
        # print(img.header.stamp)
        # print("detected boxes = ",len(object))
        box.header.seq = img.header.seq
        box.header.stamp = img.header.stamp
        box.header.frame_id = ''
        # print(box.header.stamp)
        box.source_img = img
        if len(object) != 0 and object[0].confidence > conf_thres:
            # print(object[0].bounding_box, object[0].confidence)
            box.bbox.center.x = object[0].bounding_box[0]
            box.bbox.center.y = object[0].bounding_box[1]
            box.bbox.center.theta = 0
            box.bbox.size_x = object[0].bounding_box[2]
            box.bbox.size_y = object[0].bounding_box[3]
        else:
            # print("everything -1")
            box.bbox.center.x = -1
            box.bbox.center.y = -1
            box.bbox.center.theta = -1
            box.bbox.size_x = -1
            box.bbox.size_y = -1
        #pub.publish(box)
        text_to_image = 'Processed'
        # print('Time after running detection')
        # print('Image %d' % box.source_img.header.seq)
        # print(box.source_img.header.stamp)
        
        # end = time.time()
        # print("finished callback for image", img.header.seq,"in",end-start, "seconds \n")
        #img_numpy = cv2.putText(img_numpy,text_to_image,(10,30),font, font_size, font_color, font_thickness, cv2.LINE_AA)

        # adding to time stamp log, every frame
        timelog.write('%d,%f,%f,%f,%f,%f,%f\n' % (img.header.seq,
                                                float(img.header.stamp.to_sec()),
                                                gps_t,
                                                box.bbox.center.x,
                                                box.bbox.center.y,
                                                box.bbox.size_x,
                                                box.bbox.size_y))
    # viewing/saving images
    savenum=img.header.seq
    
    
    if SAVE_IMG:
        if save_format=='.raw':
            fid = open(savedir.joinpath('Stream-%06.0f.raw' % savenum),'wb')
            fid.write(img_numpy.flatten())
            fid.close()
        elif save_format == '.avi':
            video.write(img_numpy)
        else:
            img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
            imageio.imsave(savedir.joinpath('Stream-%06.0f.jpg' % savenum), img_bgr)
    if VIEW_IMG:
        '''
        result = img_numpy
        scale_percent = 25 # percent of original size
        width = int(result.shape[1] * scale_percent / 100)
        height = int(result.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('Detection',resized)
        '''
        cv2.imshow('Detection',img_numpy)
        cv2.waitKey(1)  # 1 millisecond



def time_callback(gpstime):
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())



def init_detection_node():
    global video,timelog
    '''
    global pub,box,video,timelog
    pub = rospy.Publisher('/bounding_box', Detection2D, queue_size=1)
    box = Detection2D()

    # Initialize detection code before subscriber because this takes some time
    global imgsz, model, device, names
    print('Initializing YOLO model')
    
    if target_name == 'smoke':
        if engine:
            weights=YOLOv5_ROOT / 'smoke_BW_new_1-3-352-448.engine'
        else:
            # weights=YOLOv5_ROOT / 'smoke_BW_new.pt'
            weights=YOLOv5_ROOT / 'smoke.pt'
            # weights = 'smoke_BW_new.pt'
    else:
        if engine:
            weights=YOLOv5_ROOT / 'yolov5s_1-3-352-448.engine'
        else:
            weights=YOLOv5_ROOT / 'yolov5s.pt'
    model, device, names = detect_init(weights)
    # if engine:
    imgsz = [352,448] # scaled image size to run inference on
    # else:
    # imgsz = 640
    # imgsz = [512,640]
    # model(torch.zeros(3, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    '''

    # initializing video file
    if save_format=='.avi':
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        video = cv2.VideoWriter(str(savedir.joinpath('stream'+save_format)),
            fourcc=codec,
            fps=20,
            frameSize = (640,480)) # this size is specific to GoPro
    

    # initializing timelog
    timelog = open(savedir.joinpath('Metadata.csv'),'w')
    timelog.write('FrameID,Timestamp_Jetson,Timestamp_GPS,Center_x,Center_y,Width,Height\n')

    # initializing node
    rospy.init_node('stream_node', anonymous=False)
    #rospy.Subscriber('front_centre_cam', Image, imagecallback)
    rospy.Subscriber('/camera/image', Image, imagecallback)
    rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)
    

    rospy.spin()



'''
def detection(img0,imgsz,model,device,names,savenum):
    

    max_det=100  # maximum detections per image

    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS

    stride = model.stride
    
    t1 = time.time()
    
    # Padded resize
    img = letterbox(img0, new_shape=imgsz,stride=stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # img = np.array((img,img,img)) # not sure why this was done
    img = np.array([img])
    img = np.ascontiguousarray(img)
    # imgsz = img.shape
    seen = 0
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    # print(img.shape)
    # print('preprocessing took',(time.time()-t1)*1e3)
    
    t1 = time.time()
    #pred = model(img, augment=augment, visualize=visualize)[0]
    # replace above line with 
    pred = model(img) # not sure if it needs [0]
    # print('Inference took',1e3*(time.time()-t1))
    #NMS
    t1 = time.time()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # pred = non_max_suppression(pred,conf_thres,iou_thres)
    # print('NMS took',1e3*(time.time()-t1))
    obj = [] # initializing output list
    # Process predictions
    t1 = time.time()
    for i, det in enumerate(pred):  # per image
        seen += 1
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        annotator = Annotator(img0, line_width=1, example=str(names))
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                # print(cls)
                # extracting bounding box, confidence level, and name for each object
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                confidence = float(conf)
                object_class = names[int(cls)]
                # print(object_class)
                # if save_img or save_crop or view_img:  # Add bbox to image
                if VIEW_IMG:
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                # adding object to list
                obj.append(DetectedObject(np.array(xywh),confidence,object_class))

        im_with_boxes = annotator.result()


    #------return object with max confidence------------#
    if max_det != 1:
        bestobject = []
        bestconf = 0
        for ob in obj:
            if ob.object_class == target_name and ob.confidence > bestconf:
            # print("object class",ob.object_class)
            # if ob.object_class == 'class0' and ob.confidence > bestconf: # tensorrt loses the names
                bestobject = [ob]
                bestconf = ob.confidence  
    else:
        bestobject = [ob]
        bestconf = ob.confidence
    # print('Postprocessing took',1e3*(time.time()-t1))
    # print(bestconf)
    return bestobject,im_with_boxes


## methods from yolov5/detect_fun.py
def detect_init(weights=YOLOv5_ROOT / 'yolov5s.pt'):
    
    #device = select_device(device='',batch_size=None)   # usually cpu or cuda
    #w = str(weights[0] if isinstance(weights, list) else weights) # model weights, from .pt file
    #model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device) # initializing model in torch
    #model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights) # initializing model in torch
    #names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
	# Load model
    # device='cuda:0'
    device='cpu'
    device = select_device(device)
    
    if not engine:

        model = DetectMultiBackend(weights)    # first loads to cpu
        if half:
            model.half()    # then converts to half
        model.to(device)    # then sends to GPU
    else:
        model = DetectMultiBackend(weights, device=device)

    stride, names, pt = model.stride, model.names, model.pt
    return model, device, names
'''



class DetectedObject():
    def __init__(self,xywh=[],conf=0.,cls=''):
        self.bounding_box = xywh # normalized coordinates (by original image dimensions) of horizontal center, vertical center, width, height
        self.confidence=float(conf) # 0 is no confidence, 1 is full
        self.object_class = str(cls) # name of object (e.g., "person", or "smoke")


if __name__ == '__main__':

    try:
        init_detection_node()
    except rospy.ROSInterruptException:
        pass