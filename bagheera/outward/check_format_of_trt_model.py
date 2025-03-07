import tensorrt as trt



def is_int8_engine(engine_file_path):

    with open(engine_file_path, "rb") as f:

        runtime = trt.Runtime(trt.Logger(trt.Logger.VERBOSE))

        engine = runtime.deserialize_cuda_engine(f.read())

        inspector = engine.create_engine_inspector()

        for layer_index in range(engine.num_io_tensors):

            layer_info = inspector.get_layer_information(layer_index, trt.LayerInformationFormat.JSON)

            print(layer_info)

    return False



# Example usage

engine_path = "/home/ubuntu/outward_model_and_kpi_files/models/tycho_yolo/trt_models/int8_trt10_model.engine"

if is_int8_engine(engine_path):

    print("Model is in INT8 precision")

else:

    print("Model is not in INT8 precision")
