import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = 1 << 20
    builder.max_batch_size = 1
    builder.fp16_mode=True
    with open('/models/run03/jetracer.onnx', 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
    config.max_workspace_size = 1 << 20 
    engine = builder.build_engine(network, config)


with open('/models/run03/jetracer.engine', 'wb') as f:
    f.write(engine.serialize())
