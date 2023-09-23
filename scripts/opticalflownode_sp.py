#!/usr/bin/python3 
# # license removed for brevity
import cv2 as cv
import numpy as np
from pathlib import Path
import rospy
from rospy.client import init_node
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D,Detection2D
from sensor_msgs.msg import TimeReference
from scipy.ndimage import gaussian_filter
import os, datetime, time
import re
from kmeans_pytorch import kmeans
from pykeops.torch import LazyTensor
#------------OPTION TO TURN OFF OPTICAL FLOW-------#
OPT_FLOW_OFF = False


#---------------------------------------------------#

#--------OPTION TO VIEW FLOW RESULTS IN REAL_TIME-------------#
VIEW_IMG=True # also option to save image of output
DRAW_FLOW = True
SAVE_FLOW = False
DEBUG=False
#-----------------------------------------------------#

#--------OPTION FOR RAFT OPTICAL FLOW------------#
USE_RAFT=True # also option to save image of output
USE_COMP = True    # use any background compensation at all
USE_OUTSIDE_MEDIAN_COMP = False
USE_INPAINTING_COMP = True # more precise (theoretically) background compensation within box
USE_SEGMENTATION = True
# USE_THRESHOLDING = True
# USE_SEGMENTATION = False
USE_PADDING = True # needed for inpainting to work reliably
USE_FILTER_FLOW = False # filters out small flow values
USE_FILTER_COLOR = False # turn this False if not tracking smoke
USE_HOMOGRAPHY = False
USE_UNCERTAINTY = False
USE_MIN_VECTORS_FILTER = False
if USE_HOMOGRAPHY: USE_OUTSIDE_MEDIAN_COMP=False
UPSCALE = False
#-----------------------------------------------------#

#-----------------------------------------------------#
print_outputs = False 
#-----------------------------------------------------#

if USE_RAFT:
    # DEVICE = 'cuda'
    DEVICE = 'cpu'
    import sys
    FILE = Path(__file__).resolve()
    RAFT_ROOT = Path(FILE.parents[1] / 'src/modules/RAFT')  # RAFT directory
    if str(RAFT_ROOT) not in sys.path:
        sys.path.append(str(RAFT_ROOT))  # add RAFT_ROOT to PATH
    # sys.path.append('/home/ffil/gaia-feedback-control/src/GAIA-drone-control/src/RAFTcore')
    from raft import RAFT
    from utils.flow_viz import flow_to_image
    from utils.utils import InputPadder
    import torch
    import argparse

gps_t = 0
# create saving directory
# username = os.getlogin( )
tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))
# maindir = Path('/home/%s/1FeedbackControl' % username)
maindir = Path('./SavedData')
runs_today = list(maindir.glob('*%s*_flow' % stamp))
if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\d\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    new_run_num = max(runs_today)+1
else:
    new_run_num = 1
savedir = maindir.joinpath('%s_run%02d_flow' % (stamp,new_run_num))
os.makedirs(savedir) 


# initializing timelog
timelog = open(savedir.joinpath('Metadata.csv'),'w')
timelog.write('Frame1_ID,Timestamp_Jetson,Frame2_ID,Timestamp_Jetson,Timestamp_GPS\n')


global flowpub,flow,model_raft

skip=5
last_flow_time = 0

# def threshold(img):
#     # im1 = img[:,:,0]
#     # im2 = img[:,:,1]
#     # bg_flow = np.concatenate((im1, im2),axis=0)
#     # scale_percent = 25 # percent of original size
#     # width = int(bg_flow.shape[1] * scale_percent / 100)
#     # height = int(bg_flow.shape[0] * scale_percent / 100)
#     # dim = (width, height)
#     # cv.imshow('2 channels',cv.resize(bg_flow, dim, interpolation = cv.INTER_AREA))
#     # cv.waitKey(1)
#     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # print("Thresholding img with shape", img.shape)
#     _, thresholded_img = cv.threshold(img,200,255,cv.THRESH_BINARY)
#     # cv.imshow("thresh", thresholded_img)
#     # cv.waitKey(10)
#     return thresholded_img
#     # return bg_flow


def loopcallback(data):
    global datalist
    global timelog,last_flow_time
    if rospy.Time.now() - data.source_img.header.stamp > rospy.Duration(5):
        print("OpticalFlowNode: one of images is too old, dropping\n")
        return
    if len(datalist)==0:
        if print_outputs: print("len(datalist)==0")
        if data.bbox.center.x == -1:
            # going to wait until a bounding box is found
            if print_outputs: print("Exiting - data.bbox.center.x == -1")
            return
        else:
            # add image and bounding box as img1
            if print_outputs: print("data.bbox.center.x != -1")
            datalist.append(data)
    else:
        if print_outputs: print("len(datalist)!=0")
        dt = data.source_img.header.stamp - datalist[0].source_img.header.stamp
        if print_outputs: print("delta time =", dt)
        # if dt > rospy.Duration(0.5):
        # if dt < rospy.Duration(0.1):
        # if dt < rospy.Duration(0.4):

        #--------------------------srijan------------------------------------------#
        if dt < rospy.Duration(0.4) and dt > rospy.Duration(0.2):
        #--------------------------srijan------------------------------------------#

            if print_outputs: print("dt condition satisfied")
            if print_outputs: print("second image with proper delay found, passing to optical flow")
            # second image with proper delay found, passing to optical flow
            datalist.append(data)
            # try:
            # adding to time stamp log, every frame
            timelog.write('%d,%f,%d,%f,%f\n' % (datalist[0].source_img.header.seq,
                                        float(datalist[0].source_img.header.stamp.to_sec()),
                                        datalist[-1].source_img.header.seq,
                                        float(datalist[-1].source_img.header.stamp.to_sec()),
                                        gps_t))
            if print_outputs: print("Executing opticalflowmain()")
            flow = opticalflowmain()
            if print_outputs: print("DONE opticalflowmain()")
            flowpub.publish(flow)
            if print_outputs: print("published flow")
            if print_outputs: print(f'Flow dt: {float(datalist[-1].source_img.header.stamp.to_sec()) - float(datalist[0].source_img.header.stamp.to_sec())}')
            if print_outputs: print(f'Optical flow loop at {1/(time.time() - last_flow_time)} Hz')
            last_flow_time = time.time()
            # except:
            #     print('Cropped image too small for OF')
            # finally:
            datalist=[]
        else:
            #--------------------------srijan------------------------------------------#
            if print_outputs: print('Second image too late or too fast, emptying list')
            #--------------------------srijan------------------------------------------#

            # print('Second image too late, emptying list')
            # emptying the list of images to start fresh
            datalist = []

    # print("Exit loopcallback()")

def time_callback(gpstime):
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())
    # gps_t = gps_t
    
    # print(gps_t)
    # print(rospy.get_time())
    # print(time.time())

def init_flownode():
    """initializing optical flow node
    """
    global flowpub,flow,model_raft
    global datalist,good_first_image
    print('Initializing optical flow node')
    rospy.init_node('opticalflownode', anonymous=False)
    flowpub = rospy.Publisher('/flow',BoundingBox2D,queue_size=1)
    # rospy.Subscriber('mavros/time_reference',TimeReference,time_callback)
    flow = BoundingBox2D() # using this becaue not sure other ros message formats to use
    

    if USE_RAFT and not OPT_FLOW_OFF:        
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args(['--model',str(RAFT_ROOT.joinpath('raft-small.pth')),
                                    '--path','',
                                    '--small'])

        
        print('Initializing RAFT model')
        model_raft = flow_init(args)
        print('Done loading RAFT model')

    
    # initialize list of images with bounding boxes
    datalist = []
    # each time box is computed, append to list, and if there are enough images, run optical flow
    
    #----------------uncomment to turn ON optical flow------------#
    if not OPT_FLOW_OFF:
        rospy.Subscriber('/bounding_box', Detection2D, loopcallback)
        rospy.spin()
    #-----------------------------------------------#




def opticalflowmain():
    """callback functio to run once the two images and bounding boxes have been receieved

    Args:
        img1 (_type_): _description_
        img2 (_type_): _description_
        boundingbox (_type_): _description_
    """
    global datalist,flowpub,flow
    flow = BoundingBox2D()

    # get image 1
    tmp_img = datalist[0].source_img
    img1 = np.frombuffer(tmp_img.data,dtype=np.uint8).reshape(tmp_img.height,tmp_img.width,-1)
    # get image 2 after some skip (end of list)
    tmp_img = datalist[-1].source_img
    img2 = np.frombuffer(tmp_img.data,dtype=np.uint8).reshape(tmp_img.height,tmp_img.width,-1)

    #---------------------srijan-------------------------#
    # img1_bgr = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
    # cv.imshow('Image1', img1_bgr)
    # img2_bgr = cv.cvtColor(img2, cv.COLOR_RGB2BGR)
    # cv.imshow('Image2', img2_bgr)
    # cv.waitKey(10)

    downsample_scale = 0.4
    img1_downsampled = cv.resize(img1, None, fx=downsample_scale, fy=downsample_scale)
    img2_downsampled = cv.resize(img2, None, fx=downsample_scale, fy=downsample_scale)

    img1 = img1_downsampled
    img2 = img2_downsampled

    # img1_bgr = cv.cvtColor(img1_downsampled, cv.COLOR_RGB2BGR)
    # cv.imshow('Image1_downsampled', img1_bgr)
    # img2_bgr = cv.cvtColor(img2_downsampled, cv.COLOR_RGB2BGR)
    # cv.imshow('Image2_downsampled', img2_bgr)
    # cv.waitKey(10)
    #---------------------srijan-------------------------#

    # get bounding box for image 1
    bb = [datalist[0].bbox.center.x, datalist[0].bbox.center.y, datalist[0].bbox.size_x, datalist[0].bbox.size_y]
    # bounding box indices
    xx= int(bb[0]*img1.shape[1])
    w = int(bb[2]*img1.shape[1])
    yy = int(bb[1]*img1.shape[0])
    h = int(bb[3]*img1.shape[0])
    
    boundingbox_pixels = [yy-h//2, yy+h//2, xx-w//2, xx+w//2]
    
    # run optical flow analysis to get single vector
    flow_x,flow_y = opticalflowfunction(img1,img2,boundingbox_pixels,savenum=datalist[0].source_img.header.seq)

    # normalized displacements by the size of frame
    flow.size_x = flow_x/img1.shape[1]
    flow.size_y = flow_y/img1.shape[0]
    
    return flow

def opticalflowfunction(img1,img2,boundingbox,savenum):
    """running actual optical flow analysis on two images with a delay between them, using
    bounding box information

    Args:
        img1 (_type_): _description_
        img2 (_type_): _description_
        boundingbox (_type_): _description_

    Returns:
        _type_: _description_
    """
    if USE_RAFT:
        global model_raft
    # print('computing optical flow')
    t1 = time.time()
    color_filter_thresh = 180
    median_skipping = 1

    # extraacting bounding box indices
    y1,y2,x1,x2 = boundingbox
    # computing flow outside the bounding box
    time_init = time.time()


    if y2-y1>0.1*img1.shape[0] and x2-x1>0.1*img1.shape[1]:
        if USE_COMP:
            # im1 = threshold(img1)
            # im2 = threshold(img2)
            # image1 = cv.bitwise_and(img1, img1, mask=im1)
            # image2 = cv.bitwise_and(img2, img2, mask=im2)
            # thresh = np.concatenate((image1, image2),axis=0)
            # scale_percent = 25 # percent of original size
            # width = int(thresh.shape[1] * scale_percent / 100)
            # height = int(thresh.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # cv.imshow('Thresholded',cv.resize(thresh, dim, interpolation = cv.INTER_AREA))
            # # cv.imshow('masked',thresh)
            # cv.waitKey(1)
            if USE_RAFT:
                flow_full,_,_ = RAFTflow(img1.copy(),img2.copy())
                # cv.imshow('masked',cv.bitwise_and(img1, img1, mask=im1))
                # cv.waitKey(1)
                # flow_full,_,_ = RAFTflow(image1[y1:y2,x1:x2].copy(),image2[y1:y2,x1:x2].copy())
            else:
                flow_full = cv.optflow.calcOpticalFlowDenseRLOF(img1,img2,None) # using defaults for now
            if DEBUG: print('RAFT took %f seconds' % (time.time() - time_init))
            flow_inside = flow_full[y1:y2,x1:x2,:].copy()
            
            # segmenting image with smoke/non-smoke using flow

            # elif USE_SEGMENTATION:
            if USE_SEGMENTATION:
                t2 = time.time()
                # flow_img = flow_to_image(flow_full)
                t1 = time.time()
                # if USE_THRESHOLDING:
                #     labels = threshold(flow_full)
                # else:
                labels = flow_segment(flow_full)
                # print("LABELS SHAPE =",labels.shape, flow_full.shape)
                if DEBUG: print(f"Segmentation took {time.time()-t1} seconds")
                # determine which group is smoke
                # inside = labels[y1:y2,x1:x2].flatten()
                # outside = labels.flatten
                t5 = time.time()
                onesfrac = np.sum(labels[y1:y2,x1:x2].flatten())/np.sum(labels.flatten())
                zerosfrac = np.sum(labels[y1:y2,x1:x2].flatten()==0)/np.sum(labels.flatten()==0)
                if DEBUG: print(f"Inside/outside took {time.time()-t5} seconds")

                if np.isnan(onesfrac):
                    # this means the segmentation failed
                    if print_outputs == True: print("Segmentation Failed")
                    flow_outside = flow_full.copy()
                    flow_outside[y1:y2,x1:x2,:] = np.nan # interpolate the entire box, not just smoke
                    # labels = 1-labels # so that later its not assumed teh whole box is non-smoke
                    # remove_small_flow = True
                else:
                    # remove_small_flow = False
                    if onesfrac < zerosfrac:
                        labels = 1-labels
                    
                    # expand mask
                    t3 = time.time()
                    labels_exp = 1-gaussian_filter(1-labels,sigma=3)
                    if DEBUG: print(f"Expanding mask took {time.time()-t3} seconds")

                    t4 = time.time()
                    flow_outside = flow_full.copy()*(1-np.stack((labels_exp,labels_exp),axis=2))
                    flow_outside[flow_outside==0] = np.nan
                    if DEBUG: print(f"Flow outside assigning segment took {time.time()-t4} seconds")

                if DEBUG: print(f"Segmentation total took {time.time()-t2} seconds")
            else:
                flow_outside = flow_full.copy()
                flow_outside[y1:y2,x1:x2,:] = np.nan
        else:
            if USE_RAFT:
                flow_inside,_,_ = RAFTflow(img1[y1:y2,x1:x2,:].copy(),img2[y1:y2,x1:x2,:].copy())
                # y1 = 
            else:
                flow_inside = cv.optflow.calcOpticalFlowDenseRLOF(img1[y1:y2,x1:x2,:],img2[y1:y2,x1:x2,:],None) # using defaults for now
    else:
        return 0,0
    

        # flow_outside_x = np.nanmedian(flow_outside[:,:,0].flatten()[::median_skipping])
    # flow_outside_y = np.nanmedian(flow_outside[:,:,1].flatten()[::median_skipping])

    # # compensating for external motion using flow outside bounding box
    # flow_inside[:,:,0] -= flow_outside_x
    # flow_inside[:,:,1] -= flow_outside_y

    # # filter out low displacements
    # flow_inside[np.abs(flow_inside) < 2] = np.nan

    # # filter out if not very white (questionable approach, not likely to be generalizable but ok for now)
    # color_filter = np.mean(img1[y1:y2,x1:x2,:],axis=2) < color_filter_thresh
    # flow_inside[color_filter] = np.nan
    if USE_COMP:
        if USE_INPAINTING_COMP:

            tmp = time.time()


            
            if USE_PADDING:
                pad = 20
                u1 = np.pad(flow_full[:,:,0],pad,'reflect',reflect_type='even')
                u2 = np.pad(flow_full[:,:,1],pad,'reflect',reflect_type='even')
                u1[pad:-pad,pad:-pad] = flow_outside[:,:,0]
                u2[pad:-pad,pad:-pad] = flow_outside[:,:,1]
                padded_size = u1.shape
            else:
                u1 = flow_outside[:,:,0].copy()
                u2 = flow_outside[:,:,1].copy()
            if DEBUG: print('Padding time: %f' % (time.time()-tmp))
            
            rescale = 4
            u1 = cv.resize(u1,(u1.shape[1]//rescale,u1.shape[0]//rescale))
            u2 = cv.resize(u2,(u2.shape[1]//rescale,u2.shape[0]//rescale))

            tmp = time.time()
            # u1 = cv.inpaint(u1,(1*np.isnan(u1)).astype(np.uint8),inpaintRadius=10,flags=cv.INPAINT_TELEA)
            # u2 = cv.inpaint(u2,(1*np.isnan(u2)).astype(np.uint8),inpaintRadius=10,flags=cv.INPAINT_TELEA)
            u1 = cv.inpaint(u1,(1*np.isnan(u1)).astype(np.uint8),inpaintRadius=5,flags=cv.INPAINT_NS)
            u2 = cv.inpaint(u2,(1*np.isnan(u2)).astype(np.uint8),inpaintRadius=5,flags=cv.INPAINT_NS)
            if DEBUG: print('Inpainting time: %f' % (time.time()-tmp))
            if USE_PADDING:
                u1 = cv.resize(u1,(padded_size[1],padded_size[0]))[pad:-pad,pad:-pad]
                u2 = cv.resize(u2,(padded_size[1],padded_size[0]))[pad:-pad,pad:-pad]
            else:
                u1 = cv.resize(u1,(img1.shape[1],img1.shape[0]))
                u2 = cv.resize(u2,(img1.shape[1],img1.shape[0]))



            if DEBUG: print('Inpainting total time: %f' % (time.time()-tmp))


            flow_inside[:,:,0] -=u1[y1:y2,x1:x2]
            flow_inside[:,:,1] -=u2[y1:y2,x1:x2]
            if USE_SEGMENTATION:
                # masking out non-smoke
                flow_inside[np.stack((labels[y1:y2,x1:x2],labels[y1:y2,x1:x2]),axis=2)==0] = np.nan
                    
            flow_outside_x = np.nanmean(u1[y1:y2,x1:x2].flatten())
            flow_outside_y = np.nanmean(u2[y1:y2,x1:x2].flatten())
            if np.isnan(flow_outside_x):
                flow_outside_x = flow_outside_y = 0


            if DEBUG:
                fo = flow_outside.copy()
                fo[np.isnan(fo)]=0
                org = flow_to_image(flow_full)
                # org = cv.rectangle(org,(x1,y1),(x2,y2),color=[0,0,255],thickness=4)
                
                old = flow_to_image(fo)
                # old = cv.rectangle(old,(x1,y1),(x2,y2),color=[0,0,255],thickness=4)
                
                new = flow_to_image(np.stack((u1,u2),axis=2))
                # new = cv.rectangle(new,(x1,y1),(x2,y2),color=[0,0,255],thickness=4)
                
                bg_flow = np.concatenate((org,old,new),axis=0)
                scale_percent = 25 # changing size to % of original size
                width = int(bg_flow.shape[1] * scale_percent / 100)
                height = int(bg_flow.shape[0] * scale_percent / 100)
                dim = (width, height)
                #-------------------------------srijan---------------------------------------------------#
                # cv.imshow('background subtraction',cv.resize(bg_flow, dim, interpolation = cv.INTER_AREA))
                # cv.waitKey(10)
                #-------------------------------srijan---------------------------------------------------#

        elif USE_OUTSIDE_MEDIAN_COMP:
            

            flow_outside_x = np.nanmedian(flow_outside[:,:,0].flatten()[::1])
            flow_outside_y = np.nanmedian(flow_outside[:,:,1].flatten()[::1])
            if np.isnan(flow_outside_x):
                flow_outside_x = flow_outside_y = 0
            flow_inside[:,:,0] -= flow_outside_x
            flow_inside[:,:,1] -= flow_outside_y
    else:
        flow_outside_x = 0
        flow_outside_y = 0
    


    if USE_FILTER_FLOW:
        # filter out low displacements
        flow_inside[np.abs(flow_inside) < 5] = np.nan

    if USE_FILTER_COLOR:
        if any(~np.isnan(flow_inside.flatten())):
            tmp = time.time()
            # filter out if not very white
            # if USE_RAFT:
            #     color_filter = np.mean(prev_small,axis=2) < 180
            # else:
            color_filter = np.mean(img1[y1:y2,x1:x2,:],axis=2) < 180
            flow_inside[color_filter] = np.nan
            # print('COlor filter time: %f' % (time.time()-tmp))
    
    if USE_UNCERTAINTY:
        if any(~np.isnan(flow_inside.flatten())):
            tmp = time.time()
            sigma_inside = np.array([np.nanstd(flow_inside[:,:,0].flatten()),np.nanstd(flow_inside[:,:,1].flatten())])
            mean_inside = np.abs(np.array([np.nanmean(flow_inside[:,:,0].flatten()),np.nanmean(flow_inside[:,:,1].flatten())]))
            # print('Sigma: (%0.3f,%0.3f); Mean: (%0.3f,%0.3f)' % (sigma_inside[0],sigma_inside[1],mean_inside[0],mean_inside[1]))
            if all(sigma_inside > 2*mean_inside):
                flow_inside = np.full_like(flow_inside,np.nan)
                # print('not good')
            # print('Uncertainty time: %f' % (time.time()-tmp))
    
        # plot_flow(img1.copy(),flow_inside,flow_outside,[x1,x2,y1,y2],plot_outside_arrow=False)
    if USE_MIN_VECTORS_FILTER:
        if any(~np.isnan(flow_inside.flatten())):
            tmp = time.time()
            count = np.sum(~np.isnan(flow_inside.flatten()))/2
            # print('Count: %d' % int(count))
            BLACK = (265,265,265)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_size = 1
            font_color = BLACK
            font_thickness = 2
            img1 = cv.putText(img1,'Vectors: %d' % count,(10,img1.shape[0]-30),font, font_size, font_color, font_thickness, cv.LINE_AA)
            if count < 1e4:
                flow_inside = np.full_like(flow_inside,np.nan)
            if print_outputs == True: print('Vector count time: %f' % (time.time()-tmp))
    
    flow_inside[np.isposinf(flow_inside)] = np.nan
    flow_inside[np.isneginf(flow_inside)] = np.nan
    
    if any(~np.isnan(flow_inside.flatten())):
        boxflow_x,boxflow_y = np.nanmedian(flow_inside[:,:,0].flatten()),np.nanmedian(flow_inside[:,:,1].flatten())
    else:
        boxflow_x = boxflow_y = 0
    t2 = time.time()
    if DEBUG: print('Optical flow computation took %f seconds' % (t2-time_init))
    
    # drawing plot
    if DRAW_FLOW:
        tmp = img1.copy()
        # drawing flow arrows
        step = 5
        for ii in range(0,flow_inside.shape[1],step):
            for jj in range(0,flow_inside.shape[0],step):
                if not any(np.isnan(flow_inside[jj,ii,:])):
                    pt1 = np.array([ii,jj]) + np.array([x1,y1])
                    pt2 = np.array([ii,jj]) + np.array([x1,y1]) + np.array([flow_inside[jj,ii,0],flow_inside[jj,ii,1]]).astype(int)
                    tmp = cv.arrowedLine(
                        tmp,
                        pt1=tuple(pt1),
                        pt2=tuple(pt2),
                        color=[0,255,0],
                        thickness=1,
                        tipLength = 0.5
                    )
        
        
        if np.isnan(boxflow_x):
            boxflow_x = 0
        if np.isnan(boxflow_y):
            boxflow_y = 0
        # adding bounding box
        tmp = cv.rectangle(tmp,(x1,y1),(x2,y2),color=[0,0,255],thickness=4)


        # print(boxflow_x,boxflow_y)
        if boxflow_x != 0 and boxflow_y !=0:# drawing bulk motion arrow in bounding box
            going_to = np.array([(x1+x2)/2,(y1+y2)/2],dtype=np.uint32) + 10*np.array([boxflow_x,boxflow_y],dtype=np.uint32)
            middle = np.array([(x1+x2)//2,(y1+y2)//2],dtype=np.uint32)
            going_to[0] = min(max(0,going_to[0]),tmp.shape[1])
            going_to[1] = min(max(0,going_to[1]),tmp.shape[0])
            tmp = cv.arrowedLine(
                tmp,
                pt1 = tuple(middle),
                pt2 = tuple(going_to),
                color=[255,0,0],
                thickness=5,
                tipLength = 0.5
            )

        # drawing bulk motion arrow from entire frame
        pt1 = np.array([img1.shape[1]//2,img1.shape[0]//2],dtype=np.uint32)
        pt2 = np.array([img1.shape[1]//2,img1.shape[0]//2],dtype=np.uint32) + np.array([flow_outside_x,flow_outside_y],dtype=np.uint32)
            
        tmp = cv.arrowedLine(
            tmp,
            pt1 = tuple(pt1),
            pt2 = tuple(pt2),
            color=[255,255,0],
            thickness=5,
            tipLength=0.5
        )
        # print("Direction of flow ->", pt2-pt1)
        # plt.imshow(cv.cvtColor(tmp,cv.COLOR_BGR2RGB))
        # plt.show()
    result = tmp

    if VIEW_IMG:
        # cv.imshow('Optical Flow',result)
        scale_percent = 25 # percent of original size
        width = int(result.shape[1] * scale_percent / 100)
        height = int(result.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv.resize(result, dim, interpolation = cv.INTER_AREA)
        cv.imshow('Optical Flow',result)
        #  cv.imshow('Optical Flow',resized)
        cv.waitKey(1)
        if DEBUG: print('Optical flow plotting time %f' % (t2-t1))


        
    if SAVE_FLOW:
        t1 = time.time()
        savename = savedir.joinpath('OpticalFlow-%06.0f.jpg' % savenum)
        cv.imwrite(str(savename),tmp)
        np.savez(savedir.joinpath('OpticalFlow-%06.0f' % savenum),
                                    flow_outside_x=flow_outside_x,
                                    flow_outside_y=flow_outside_y,
                                    flow_inside=flow_inside,
                                    boxflow_x=boxflow_x,
                                    boxflow_y=boxflow_y,
                                    bbox=[x1,x2,y1,y2])
        t2 = time.time()
        # print('Optical flow saving time %f' % (t2-t1))
        # if VIEW_IMG:

    
    
    return np.float64(boxflow_x),np.float64(boxflow_y)

def RAFTflow(img1,img2):
    global model_raft
    # print('Image size before pad:')
    # print(img1.shape)
    #if UPSCALE:
        #img1 = cv.resize(img1,(img1.shape[1]*2,img1.shape[0]*2))
        #img2 = cv.resize(img2,(img1.shape[1]*2,img1.shape[0]*2))
        
    with torch.no_grad():

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img1 = img1[None].to(DEVICE)
        img2 = img2[None].to(DEVICE)

        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        # print(img1.size())
        _,flow = model_raft(img1,img2,iters=20,test_mode=True)
        # print("FLOW SHAPE", flow.shape)
        flow = padder.unpad(flow)
        flow = flow[0].permute(1,2,0).cpu().numpy()
        # print('Flow size after unpad:')
        # print(flow.shape)
        img1 = img1[0].permute(1,2,0).cpu().numpy()
        img2 = img2[0].permute(1,2,0).cpu().numpy()
    if UPSCALE:
        return flow[::2,::2,:]/2,img1[::2,::2,:],img2[::2,::2,:]
    else:
        return flow,img1,img2
   
def flow_init(args):
    global model_raft
    model_raft = torch.nn.DataParallel(RAFT(args))
    model_raft.load_state_dict(torch.load(args.model))

    model_raft = model_raft.module
    model_raft.to(DEVICE)
    model_raft.eval()
    if print_outputs == True: print('Warming up...')
    x,y = np.meshgrid(np.linspace(0,100,640),np.linspace(0,100,480))
    img1 = np.sum(np.stack((x,y),2),axis=2)
    img2 = img1+50
    img1 = cv.cvtColor(img1.astype(np.uint8),cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2.astype(np.uint8),cv.COLOR_GRAY2BGR)
# cv.imshow('stack',np.concatenate((img1,img2),axis=1))
# cv.waitKey(0)
    RAFTflow(img1,img2)
    return model_raft

def flow_segment(flow_img):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = flow_img.reshape((-1, flow_img.shape[2]))
    # convert to float
    pixel_values = np.float32(pixel_values)


    #-------OPENCV METHOD WITH CPU--------------#
    # define stopping criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, 0.9)
    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(flow_img.shape[:-1])
    #--------------------------------------------#

    # OR

    #---------PYKEOPS METHOD WITH GPU---------------#
    # pixel_values = np.ascontiguousarray(pixel_values)
    # pixel_values = torch.from_numpy(pixel_values)
    # pixel_values = pixel_values.float()

    # # kmeans
    # # k = 2
    # # labels, cluster_centers = kmeans(
    # #     X=pixel_values, num_clusters=k, distance='euclidean', device=torch.device(DEVICE)
    # # )

    # labels, c = KMeans(pixel_values, K=2,use_cuda=DEVICE=='cuda',verbose=DEBUG)
    
    # labels = labels.reshape(flow_img.shape[:-1]).cpu().numpy()
    #--------------------------------------------#
    return labels

def KMeans(x, K=2, Niter=10, use_cuda=True,verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        if print_outputs == True: print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        if print_outputs == True: print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c



if __name__ == '__main__':
    try:
        init_flownode()
    except rospy.ROSInterruptException:
        pass

