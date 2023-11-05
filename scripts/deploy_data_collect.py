#!/usr/local/bin/python3.10

import msgpackrpc
import rospy
import airsim
import numpy as np
from sensor_msgs.msg import Image
import cv2
import os, re
from pathlib import Path
import sys, datetime
from std_msgs.msg import String
HOST = '127.0.0.1' # Standard loopback interface address (localhost)
from platform import uname
if 'linux' in uname().system.lower() and 'microsoft' in uname().release.lower(): # In WSL2
    if 'WSL_HOST_IP' in os.environ:
        HOST = os.environ['WSL_HOST_IP']
print("Using WSL2 Host IP address: ", HOST)
# client = airSim.multiiRotorClient(ip=HOST)
client = airsim.MultirotorClient(ip=HOST)
client.confirmConnection()
client.enableApiControl(True)


global view_img, save_img, save_format
view_img = True
save_img = False
save_format = '.png'

tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))
maindir = Path('./SavedData')
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



def saveimagecallback(smoketrack):
    global view_img, save_img, save_format
    if smoketrack != None:
        save_img = True
        print('Saving Images')


def init_data_collection_node():
    global view_img, save_img, save_format
    rospy.init_node("droneCam_datacollect")
    rospy.Subscriber('smoketrack', String, saveimagecallback)
    seq = 1
    img = Image()

    while not rospy.is_shutdown():
        responses = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
        response = responses[0]

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        img.header.seq = seq
        img.header.stamp = rospy.Time.now()
        seq += 1
        img.height,img.width = response.height, response.width
        img.data = img_rgb.flatten().tolist()

        rospy.sleep(0.0001)

        if view_img:
            result = img_rgb
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
                cv2.imwrite(filename,img_rgb)


if __name__ == '__main__':
    try:
        init_data_collection_node()
    except rospy.ROSInterruptException:
        pass