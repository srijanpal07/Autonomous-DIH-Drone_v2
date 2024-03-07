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
from std_msgs.msg import String
from sensor_msgs.msg import TimeReference


global view_img, save_img, save_format
view_img = True
save_img = True
save_format = '.png'

tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))
maindir = Path('~/Data')
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


def time_callback(gpstime):
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())


def saveimagecallback(smoketrack):
    global view_img, save_img, save_format
    if smoketrack != None:
        save_img = True
        print('Saving Images')


def imagecallback(img):
    global pub,box,video,timelog
    global imgsz, model, device, names
    # box = Detection2D()
    img_numpy = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)

    if rospy.Time.now() - img.header.stamp > rospy.Duration(0.5):
        print("DetectionNode: dropping old image from detection\n")
        # text_to_image = 'skipped'
        return
    else:
        print('DetectionNode: Running detection inference')        
        
        if view_img:
            result = img_numpy
            scale_percent = 25 # percent of original size
            width = int(result.shape[1] * scale_percent / 100)
            height = int(result.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('Streamed images ',resized)
            cv2.waitKey(1) 

        if save_img:
            savenum = img.header.seq
            if save_format=='.png':
                filename = str(savedir.joinpath(f'stream{tmp.month:02d}{tmp.day:02d}{tmp.year}_{savenum:06d}.png'))
                cv2.imwrite(filename,img_numpy)
    # viewing/saving images
    savenum=img.header.seq



def init_data_collection_node():
    global view_img, save_img, save_format
    rospy.init_node("droneCam_datacollect")
    rospy.Subscriber('smoketrack', String, saveimagecallback)
    rospy.Subscriber('/camera/image', Image, imagecallback)
    rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)
    seq = 1
    img = Image()

    while not rospy.is_shutdown():
        rospy.sleep(0.000)


if __name__ == '__main__':
    try:
        init_data_collection_node()
    except rospy.ROSInterruptException:
        pass