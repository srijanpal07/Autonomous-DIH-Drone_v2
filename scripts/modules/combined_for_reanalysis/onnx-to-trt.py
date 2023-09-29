import torch
import tensorrt as trt
import onnx

"""https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics
    https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/

    need to add trt.init_libnvinfer_plugins(None,'') before the deserialization in whatever code used to run the engine later
    https://github.com/onnx/onnx-tensorrt/issues/514
"""

def main():
    modelname = 'smoke'

    batches = 1
    channels = 3
    hh = 352
    ww = 448

    onnx_path = '%s_%d-%d-%d-%d.onnx' % (modelname,batches,channels,hh,ww)
    trt_path = '%s_%d-%d-%d-%d.engine' % (modelname,batches,channels,hh,ww)

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
    builder.max_batch_size = 8
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


if __name__=='__main__':
    main()