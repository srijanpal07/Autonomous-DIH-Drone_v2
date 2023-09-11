    # laoding model manually
import torch
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from models.common import DetectMultiBackend
from models.yolo import DetectionModel
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.augmentations import letterbox


def main():
    # weights = 'yolov5s.pt'
    # weights = 'yolov5s_1-3-352-448.engine'
    # weights = 'smoke.pt'
    weights = 'smoke_1-3-352-448.engine'
    batch = 1
    imgsz = (448, 352) # width, height for model

    # device=torch.device('cpu')
    device = 'cuda:0'
    # img = cv.imread('data/images/zidane.jpg')
    img = cv.imread('data/images/smoke-grenade.jpg')

    print('preprocessing image')
    img,img0 = preprocess(img,imgsz,device,batch,half=True)

    print('loading model')
    if weights[-2:]=='pt':

        model = DetectMultiBackend(weights=weights).half().to(device)
    else:
        model = DetectMultiBackend(weights=weights)
    
    model.eval()

    print('warming up')
    # warmup
    for _ in range(2):
        y = model(img)

    # model_man = torch.load(weights)['model'].float()
    # model_man.to(device)
    # # model_man.half()

    # dummy = torch.randn(1,3,480,640)
    # dummy.half()
    # dummy.to(device)

    print('running inference')
    # y = model_man(img)
    t1 = time.time()
    yy = model(img)
    print('Inference time',1e3*(time.time()-t1))
    t2 = time.time()
    yy = non_max_suppression(yy,0.4,0.45)
    print('NMS time',1e3*(time.time()-t2))


    result = postprocess(yy,img,img0,model.names)
    print('Full time',1e3*(time.time()-t1))
    # plt.imshow(result)
    # plt.show()
    cv.imshow('result',result)
    cv.waitKey(5000)
    print('done')
    
def preprocess(img0,imgsz=(640,480),device='cpu',batchsize=1,half=True):
    
    img = cv.resize(img0,imgsz)


    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    if batchsize > 1:
        img = np.np.array((img,img,img))
    else:
        img = np.array([img])
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    # img = img.float()
    if half:
        img = img.half()
    else:
        img = img.float()
    # img = img.half() if model.fp16 else im.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    return img,img0

def postprocess(det,img,img0,names):
    obj = [] # initializing output list
    # Process predictions
    t1 = time.time()
    # for i, det in enumerate(pred):  # per image
    # seen += 1
    # img = img0.copy()
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    
    annotator = Annotator(img0, line_width=1, example=str(names))
    det = det[0]
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            # print(cls)
            # extracting bounding box, confidence level, and name for each object
            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            # confidence = float(conf)
            # object_class = names[int(cls)]
            # print(object_class)
            # if save_img or save_crop or view_img:  # Add bbox to image
            # if VIEW_IMG:
            c = int(cls)  # integer class
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
            # adding object to list
            # obj.append(DetectedObject(np.array(xywh),confidence,object_class))

    im_with_boxes = annotator.result()
    return im_with_boxes

if __name__=='__main__':
    main()