#!/usr/local/bin/python3.10

import msgpackrpc
import rospy
import airsim
import numpy as np
from sensor_msgs.msg import Image
import cv2
import os
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

# png_image = client.simGetImage("0", airsim.ImageType.Scene)
# img = airsim.string_to_uint8_array(png_image)
# # img=png_image
# img = np.reshape(img,(640,480))
rospy.init_node("droneCam")
pub = rospy.Publisher("front_centre_cam", Image, queue_size = 1)

seq = 1
img = Image()

while not rospy.is_shutdown():
# while(1):
    responses = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    # print(responses)
    # print("height=",response.height, ", width =", response.width)

    # get numpy array
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 

    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(response.height, response.width, 3)

    img.header.seq = seq
    img.header.stamp = rospy.Time.now()
    seq += 1
    img.height,img.width = response.height, response.width
    img.data = img_rgb.flatten().tolist()

    pub.publish(img)
    rospy.sleep(0.0001)
    # original image is fliped vertically
    # img_rgb = np.flipud(img_rgb)