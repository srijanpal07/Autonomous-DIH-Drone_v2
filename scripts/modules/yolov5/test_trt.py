import torch
import numpy as np
import tensorrt as trt
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression

weights = 'smoke.pt'

device='cuda:0'
device = select_device(device)

model = DetectMultiBackend(weights,device=device)


dummy_input = torch.randn(3, 3, 480, 640).to(device)

y = model(dummy_input)

yy = non_max_suppression(y)

print('done')
