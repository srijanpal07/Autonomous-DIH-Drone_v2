#!/usr/bin/python3.8
# license removed for brevity

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import TimeReference
import os, datetime
import PySpin
import sys, subprocess, time
import cv2
import queue
from pathlib import Path
sys.path.append("goproapi")
import re
import numpy as np
import time



# ------ OPTION TO VIEW ACQUISITION IN REAL_TIME ------ #
VIEW_IMG = True
SAVE_IMAGE = False
SAVE_FORMAT = False # '.avi'
# ----------------------------------------------------- #


# create saving directory
username = os.getlogin( )
tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))
maindir = Path('/home/%s/1FeedbackControl' % username)
runs_today = list(maindir.glob('*%s*_camera' % stamp))
if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\d\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    new_run_num = max(runs_today)+1
else:
    new_run_num = 1
savedir = maindir.joinpath('%s_run%02d_camera' % (stamp,new_run_num))
os.makedirs(savedir)

gps_t = 0




def time_callback(gpstime):
    """
    time callback function
    """
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())




def publishimages(cam,camlist):
    """
    This function acquires images from a device.
    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global system

    pub = rospy.Publisher('/camera/image', Image, queue_size=1)
    rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)
    rospy.init_node('cameranode', anonymous=False)

    device_serial_number = False

    # initializing timelog
    timelog = open(savedir.joinpath('Timestamps.csv'),'w')
    timelog.write('FrameID,Timestamp_Jetson,Timestamp_GPS\n')

    try:
        i = 0
        img = Image()
        while not rospy.is_shutdown():
            try:
                img_raw = cam.GetNextImage()
                if img_raw.IsIncomplete():
                    print("Image incomplete, skipping")
                    img_raw.Release()
                    continue
                else:
                    i += 1
                img_raw = img_raw.GetNDArray()
                img_raw = cv2.cvtColor(img_raw,cv2.COLOR_BayerRG2RGB)
                # img_raw = cv2.resize(img_raw,(640,480))
                ret = True
                img.header.seq = i
                img.header.stamp = rospy.Time.now()


                # adding to time stamp log
                timelog.write('%d,%f,%f\n' % (img.header.seq,float(img.header.stamp.to_sec()),gps_t))

                if not ret:
                    print('Image capture with opencv unsuccessful')
                else:
                    img.height,img.width = img_raw.shape[0], img_raw.shape[1]
                    if VIEW_IMG:
                        cv2.imshow('FLIR',img_raw)
                        cv2.waitKey(1)

                    img.data = img_raw.flatten().tolist()
                    pub.publish(img)

                    if SAVE_IMAGE:
                        # Create a unique filename
                        if device_serial_number:
                            filename = savedir.joinpath('Acquisition-%s-%06.0f' % (device_serial_number, i))
                        else:
                            filename = savedir.joinpath('Acquisition-%06.0f' % i)

                        if SAVE_FORMAT =='.raw':
                            fid = open(str(filename)+SAVE_FORMAT,'wb')
                            fid.write(img_raw.flatten())
                            fid.close()
                        elif SAVE_FORMAT == '.avi':
                            if i==1:
                                codec = cv2.VideoWriter_fourcc('M','J','P','G')
                                video = cv2.VideoWriter(str(savedir.joinpath('Acquisition'+SAVE_FORMAT)),
                                    fourcc=codec,
                                    fps=10,
                                    frameSize = (img_raw.shape[1],img_raw.shape[0]))
                            video.write(img_raw)
                        elif SAVE_FORMAT == '.jpg':
                            cv2.imwrite(str(filename)+SAVE_FORMAT, img_raw)


            except Exception as e:       
                print('Error: %s' % e)


        cam.EndAcquisition()
        cam.DeInit()
        del cam
        camlist.Clear()
        system.ReleaseInstance()

        cv2.destroyAllWindows()
    except Exception as e:
        print('Error: %s' % e)



if __name__ == '__main__':
    system = PySpin.System.GetInstance()
    camlist = system.GetCameras()
    try:
        if len(camlist)>0:
            print('Connecting to FLIR camera')
            cam = camlist.GetByIndex(0)
        else:
            raise Exception('NO CAMERAS FOUND')

        cam.Init()
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(10)  # Adjust the frame rate as needed

        # Start image acquisition
        cam.BeginAcquisition()
        cam.Width = cam.WidthMax
        cam.Height = cam.HeightMax

        publishimages(cam,camlist)
    except rospy.ROSInterruptException:
        if len(camlist)>0:
            print('Closing out camera')
            cam.close()
