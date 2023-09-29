import torch
import onnx
import tensorrt as trt
from models.experimental import attempt_load
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
# import torchvision

"""https://pytorch.org/docs/stable/onnx.html
https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
"""

def main():

    # modelname = 'raft_small'
    # pt_path = '%s.pth' % modelname
    modelname = 'yolov5s'
    pt_path = '%s.pt' % modelname
    
    device='cuda:0'
    device = select_device(device)
    # dummy_input = torch.randn(3, 3, 480, 640).to(device)
    dummy_input = torch.randn(3, 3, 352, 448).to(device)
    # dummy_input = torch.randn(1, 3, 480, 640).to(device)

    onnx_path = '%s_%d-%d-%d-%d.onnx' % (modelname,dummy_input.size(0),dummy_input.size(1),dummy_input.size(2),dummy_input.size(3))
    trt_path = '%s_%d-%d-%d-%d.engine' % (modelname,dummy_input.size(0),dummy_input.size(1),dummy_input.size(2),dummy_input.size(3))

    print('loading model')
    model = DetectMultiBackend(pt_path,device=device,fp16=True)
    

    model.half()
    model.to(device)
    model.eval()

    for _ in range(2):
        y = model(dummy_input) # dry runs needed

    print('exporting model')
    torch.onnx.export(model,dummy_input,onnx_path,input_names=["images"],output_names=["output"],opset_version=11,export_params=True)

    print('checking onnx model')
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    del onnx_model
    del model
    torch.cuda.empty_cache()

    print('Model successfully converted to ONNX, saved to %s' % onnx_path)



    print('Exporting to tensorrt')

    # device='cuda'
    TRT_LOGGER = trt.Logger()
    engine = build_engine(onnx_path,TRT_LOGGER)
    # print('serializing')
    # engine.serialize()

    print('writing to file')
    with open(trt_path, 'wb') as f:
        f.write(engine)

def build_engine(onnx_path,TRT_LOGGER):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    
    # builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # config.max_match_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        # builder.fp16_mode = True
        print('Building with FP16')
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print('Buidling with FP32')

    # parse ONNX
    with open(onnx_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        success = parser.parse(model.read())
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        if not success:
            print('could not parse!')
            
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    # engine = builder.build_engine(network,config)
    serialized_engine = builder.build_serialized_network(network, config)
    # context = engine.create_execution_context() # only needed for inference
    print("Completed creating Engine")
    # return engine, context
    # return engine
    return serialized_engine

if __name__=="__main__":
    main()
