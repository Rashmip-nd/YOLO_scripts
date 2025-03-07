import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2

import argparse
import os

def resize_and_pad(image, new_shape):
    old_size = image.shape[:2] 
    ratio = float(new_shape[-1]/max(old_size))#fix to accept also rectangular images
    new_size = tuple([int(x*ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    
    delta_w = new_shape[1] - new_size[1]
    delta_h = new_shape[0] - new_size[0]
    
    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)
    
    return new_im, delta_w, delta_h


def main(input_dir, mode, model_name, output_dir):

    print(f"=======================================================")
    print(f"intput_dir = {input_dir}")
    print(f"mode       = {mode}")
    print(f"model_name = {model_name}")
    print(f"output_dir = {output_dir}")
    print(f"=======================================================")

    # Initialize OpenVINO Runtime core
    core = ov.Core()
    # Read a model
    model = core.read_model(model_name)

    ############################################################################################################
    num_outputs = len(model.outputs)
    print(f"\n\nThe model has {num_outputs} outputs.")
    for i in range(num_outputs):
        output_name = model.output(i)
        print(f"Output name: {output_name}")
    print(f"\n\n")
    ############################################################################################################

    # Inizialize Preprocessing for the model
    ppp = PrePostProcessor(model)
    # Specify input image format
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
    # Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB)
    # Specify model's input layout
    ppp.input().model().set_layout(Layout("NCHW"))
    # Specify output results format
    ppp.output().tensor().set_element_type(Type.f32)

    # Embed above steps in the graph
    model = ppp.build()

    # Compile model for specified device
    if mode == "HDDL":
        #core.set_property(mode, {"HDDL_DEVICE_TAG":"MyTag"})
        #core.set_property(mode, {"HDDL_BIND_DEVICE":"YES"})
        #core.set_property(mode, {"HDDL_RUNTIME_PRIORITY":"1"})
        #compiled_model = core.compile_model(model, mode,  config={"LOG_LEVEL":"LOG_DEBUG"})
        compiled_model = core.compile_model(model, mode)
    else:
        compiled_model = core.compile_model(model, mode)
    
    # Create an infer request for model inference 
    infer_request = compiled_model.create_infer_request()

    # Read input images
    for filename in os.listdir(input_dir):  
        if filename.endswith(".jpg") or filename.endswith(".png"):   
            img = cv2.imread(os.path.join(input_dir, filename))

            # resize image
            img_resized, dw, dh = resize_and_pad(img, (384, 320))

            # [DEBUG CODE]
            #print(f"{os.path.join(output_dir,filename.split('.')[0])}_vis.png")
            #cv2.imwrite(os.path.join(output_dir,filename.split('.')[0]) + "_vis.png", img_resized)

            # Create tensor from image
            input_tensor = np.expand_dims(img_resized, 0)

            # [DEBUG CODE]
            # Generate a random image with dimensions 640x320x3                                                                                                                               
            #random_image = np.random.rand(384, 320, 3) * 255                                                                                                                                          
            #random_image = random_image.astype(np.uint8)
            #input_tensor = np.expand_dims(random_image, 0)   

            print(f'input_tensor shape {input_tensor.shape}')
            infer_request.infer({0: input_tensor})

            # Retrieve inference results 
            result = infer_request.get_output_tensor()
            print(f"shape of result for filename : {filename} is {result.shape}")

            # Write output boxes as numpy raw file
            resultpath = os.path.join(output_dir, f"{filename.split('.')[0]}_boxes.raw")
            result.data.tofile(resultpath)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and visualise the bboxes on output image")
    parser.add_argument(
        "-a",
        "--input_dir",
        type=str,
        help="directory with image and model",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--mode",
        type=str,
        help="name of the mode CPU or HDDL",
        required=True,
    )
    parser.add_argument("-c", "--model_name", type=str, help="name of the onnx model (without extension)", required=True,)
    parser.add_argument(
        "-d",
        "--output_dir",
        type=str,
        help="name of the prediction numpy file (without extension)",
        required=True,
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):  
        os.makedirs(args.output_dir) 

    main(args.input_dir, args.mode, args.model_name, args.output_dir)
