import torch
import onnx
# from models.experimental import attempt_load
# from models.common import DetectMultiBackend
# from utils.torch_utils import select_device
from raft import RAFT
# import torchvision
import argparse

"""https://pytorch.org/docs/stable/onnx.html
https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
"""

def main():

    modelname = 'raft-small'
    pt_path = '%s.pth' % modelname
    # modelname = 'smoke'
    # pt_path = '%s.pt' % modelname
    onnx_path = '%s.onnx' % modelname
    # onnx_path='dummy.onnx'
  
    print('loading model')
    device='cuda:0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args(['--model',pt_path,
                                '--path','',
                                '--small'])
    # device = select_device(device)
    # model = DetectMultiBackend(pt_path,device=device,fp16=True)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module

    model.half()
    model.to(device)
    model.eval()

    
    # model = attempt_load(pt_path,device=device,inplace=True,fuse=True)
    # dummy_input = torch.randn(3, 3, 480, 640).to(device)
    # dummy_input = torch.randn(1, 3, 352, 448).to(device)
    dummy_input = torch.randn(1, 3, 480, 640)
    dummy_input = dummy_input.half()
    dummy_input.to(device)
    dummy_input = (dummy_input,dummy_input)
 

    for _ in range(2):
        y = model(dummy_input) # dry runs needed

    print('exporting model')
    torch.onnx.export(model,dummy_input,onnx_path,input_names=["images"],output_names=["output"],opset_version=11,export_params=True)

    print('checking onnx model')
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print('Model successfully converted to ONNX, saved to %s' % onnx_path)

if __name__=="__main__":
    main()
