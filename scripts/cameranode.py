#!/usr/bin/env python3 
# license removed for brevity

from tkinter import image_types
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import TimeReference
import os, datetime
import sys, subprocess, time
import cv2
from goprocam import GoProCamera
from goprocam import constants
import queue, threading
from pathlib import Path
sys.path.append("goproapi")
import re
import numpy as np


#--------OPTIONS-------------#
VIEW_IMG = True
SAVE_IMAGE = True
SAVE_FORMAT = '.avi'
USE_DEWARPING = True # reduces fps if True
#----------------------------#


# create saving directory
gps_t = 0
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


# loading camera distortion parameters
FILE = Path(__file__).resolve()
cal_path = FILE.parent.joinpath('gopro_intrinsics.npz')
cal = np.load(cal_path)
intrinsics = dict(
    mtx = cal['cam_matrix'],
    dist = cal['distortion_coeff'],
    crop = 75
)

h,w = 480,640
intrinsics['newcameramtx'], intrinsics['roi'] = cv2.getOptimalNewCameraMatrix(intrinsics['mtx'], intrinsics['dist'], (w,h), 0, (w,h))



# seems to find device automatically if connected? Why don't we do this?
serialstring = 'DeviceSerialNumber'
# serialstring = '18285036'
# /home/ffil/gaia-feedback-control/src/GAIA-drone-control/src/goproapi/gopro_keepalive.py
# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

  # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            # print("-------------OK-----------------")
            ret, frame = self.cap.read()
            if not ret:
                # break
                continue
            while not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()
        return





def time_callback(gpstime):
    """
    time callback function
    """
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())



def undistort(img,params):
    """
    undistort ??
    """
    crop_pix = params['crop']
    dst = cv2.undistort(img, params['mtx'], params['dist'], None, params['newcameramtx'])
    # crop the image
    x, y, w, h = params['roi']
    dst = dst[y:y+h, x:x+w]
    # dst = dst[h//6:-h//6,w//6:-w//6]
    dst = dst[crop_pix:-crop_pix,crop_pix:-crop_pix]
    return dst



def publishimages():
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

    pub = rospy.Publisher('/camera/image', Image, queue_size=1)
    rospy.init_node('cameranode', anonymous=False)
    rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)

    capture_init()

    gpCam = GoProCamera.GoPro()
    #gpCam.gpControlSet(constants.Stream.BIT_RATE, constants.Stream.BitRate.B2_4Mbps)
    #gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.W480)
    device_serial_number = False

    # initializing timelog
    timelog = open(savedir.joinpath('Timestamps.csv'),'w')
    timelog.write('FrameID,Timestamp_Jetson,Timestamp_GPS\n')

    try:
        result = True
        
        # cap = cv2.VideoCapture("udp://127.0.0.1:10000") # stream from gopro wifi
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # setting buffer size to 1, only latest frame kept
        cap = VideoCapture("udp://127.0.0.1:10000") # stream from gopro wifi
        capt1 = 0

        # Retrieve, convert, and save images
        i = 0
        first_image = True
        img = Image()

        while not rospy.is_shutdown():
            i += 1
            t1 = time.time()
            # print("OK")
            img_raw = cap.read()
            ret = True

            # capt2 = time.time()
            # capt1 = capt2

            # if img_raw is not None:
            if ret:
                # ret = True
                if first_image:
                    print('#-------------------------------------------#')
                    print('Image capture successfully started')
                    print('#-------------------------------------------#')
                    first_image = False
                
                img.header.seq = i
                img.header.stamp = rospy.Time.now()

                # adding to time stamp log
                timelog.write('%d,%f,%f\n' % (img.header.seq,float(img.header.stamp.to_sec()),gps_t))

                if USE_DEWARPING:
                    img_raw = undistort(img_raw, intrinsics)

                img.height, img.width = img_raw.shape[:-1]

                if VIEW_IMG:
                    cv2.imshow('gopro', img_raw)
                    cv2.waitKey(1)
                
                img.data = img_raw.flatten().tolist()
                # Send image on topic
                pub.publish(img)

                if SAVE_IMAGE:
                    # Create a unique filename
                    if device_serial_number:
                        filename = savedir.joinpath('Acquisition-%s-%06.0f' % (device_serial_number, i))
                    else:  # if serial number is empty
                        filename = savedir.joinpath('Acquisition-%06.0f' % i)


                    if SAVE_FORMAT=='.raw':
                        fid = open(str(filename)+SAVE_FORMAT,'wb')
                        fid.write(img_raw.flatten())
                        fid.close()
                    elif SAVE_FORMAT == '.avi':
                        if i==1:
                            codec = cv2.VideoWriter_fourcc('M','J','P','G')
                            video = cv2.VideoWriter(str(savedir.joinpath('Acquisition'+SAVE_FORMAT)),
                                fourcc=codec,
                                fps=30,
                                frameSize = (img_raw.shape[1],img_raw.shape[0]))

                        video.write(img_raw)
                    else:
                        cv2.imwrite(str(filename)+SAVE_FORMAT,img_raw)

            else:
                cv2.imwrite(str(filename) + SAVE_FORMAT, img_raw)

        cap.release()
        # video.release()
    
    except rospy.ROSInterruptException:
        cap.release()

    except Exception as e:       
        print('Error: %s' % e)



def capture_init():
    """
    starts gopro streaming in background
    """
    FILE = Path(__file__).resolve()
    goproapi = Path(FILE.parents[1] / 'src/goproapi')  # gopro functions directory
    cmd = str(goproapi.joinpath('gopro_keepalive.py'))
    # subprocess.call(['python3',cmd])
    with open(os.devnull, 'w') as fp:
        output = subprocess.Popen('python3 ' + cmd, stdout=fp,shell=True)
    return




if __name__ == '__main__':
    try:
        publishimages()
    except rospy.ROSInterruptException:
        pass

