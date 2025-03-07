
"""
This script performs object detection using the Tycho model on images or videos.

The script includes the following functions:
- resize_and_pad: Resizes and pads an image to a specified shape.
- resize_and_mean_sub: Resizes an image to a specified shape and performs mean subtraction.
- bbox_transform_inv: Applies bounding box inverse transformation to predicted deltas.
- py_cpu_nms: Pure Python implementation of non-maximum suppression.
- get_detections: Gets the detections for model predicted boxes and scores.
- visualize_bbox: Visualizes a single bounding box on an image.
- visualize: Visualizes bounding boxes and classes on an image.
- process_image: Processes an image using the Tycho model.
- process_folder: Processes a folder of images using the Tycho model.
- process_video: Processes a video using the Tycho model.

The script also includes a dictionary `tycho_label_map` that maps class indices to class names, IDs, and thresholds.

To use the script, you can call the `process_folder` function to process images in a folder or the `process_video` function to process a video.

Example usage:
- Process a folder of images:
    process_folder('input_dir', 'output_dir', infer_request)

- Process a video:
    process_video('input.mp4', 'output_dir', infer_request)
"""

import math
import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2

import argparse
import os

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White
TARGET_FPS = 1

tycho_label_map = {
  0: { "name": "BACKGROUND", "id": "-1", "threshold": 1.0 },
  1: { "name": "CAR", "id": "1" , "threshold": 0.90 },
  2: { "name": "TRUCK" , "id": "2" , "threshold": 0.85 },
  3: { "name": "TRAFFIC_LIGHTS" , "id": "100" , "threshold": 0.80 },
  4: { "name": "TRAFFIC_SIGN" , "id": "200" , "threshold": 0.55 },
  5: { "name": "PEDESTRIAN" , "id": "20000" , "threshold": 0.85 },
  6: { "name": "CAR_FRONT" , "id": "11" , "threshold": 0.85 },
  7: { "name": "TRUCK_FRONT" , "id": "21" , "threshold": 0.75 },
  8: { "name": "ROAD_INFORMATION_SIGN" , "id": "300" , "threshold": 0.60 },
  9: { "name": "ROAD_MARKING" , "id": "400" , "threshold": 0.70 },
  10: { "name": "FIRE_HYDRANT" , "id": "500" , "threshold": 0.80 },
  11: { "name": "CONSTRUCTION_CONES" , "id": "600" , "threshold": 0.85 },
  12: { "name": "OTHER" , "id": "1000" , "threshold": 0.95 },
}

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

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

def resize_and_mean_sub(image, new_shape):
    """
    Resize the input image to the specified new shape and perform mean subtraction.

    Args:
        image (numpy.ndarray): The input image.
        new_shape (tuple): The new shape of the image in the format (height, width).

    Returns:
        numpy.ndarray: The resized and mean-subtracted image.

    """
    old_size = image.shape[:2] 
    ratio = float(new_shape[-1]/max(old_size))#fix to accept also rectangular images
    new_size = tuple([int(x*ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    image = image.astype(np.float32)
    
    mean = [102.9801, 115.9465, 122.7717]
    image[:, :, 0] -= mean[0]
    image[:, :, 1] -= mean[1]
    image[:, :, 2] -= mean[2]
    
    image = image.transpose((2, 0, 1))

    return image, ratio

def bbox_transform_inv(boxes, deltas, weights=[1.0, 1.0, 1.0, 1.0], offset=1.0):
    """
    Applies bounding box inverse transformation to predicted deltas.

    Args:
        boxes (numpy.ndarray): Array of shape (N, 4) representing the bounding boxes.
        deltas (numpy.ndarray): Array of shape (N, 4) representing the predicted deltas.
        weights (list, optional): List of length 4 representing the weights for each delta component. Defaults to [1.0, 1.0, 1.0, 1.0].
        offset (float, optional): Offset value added to the width and height of the boxes. Defaults to 1.0.

    Returns:
        numpy.ndarray: Array of shape (N, 4) representing the transformed bounding boxes in image space.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + offset
    heights = boxes[:, 3] - boxes[:, 1] + offset
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]/weights[0]
    dy = deltas[:, 1::4]/weights[1]
    dw = np.minimum(math.log(1000.0/16), deltas[:, 2::4]/weights[2])
    dh = np.minimum(math.log(1000.0/16), deltas[:, 3::4]/weights[3])

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def py_cpu_nms(dets, thresh, offset=1.0):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + offset) * (y2 - y1 + offset)
    # order = scores.argsort()[::-1]
    # commented to match with cpp implementation of nms
    order = np.arange(0, len(scores))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + offset)
        h = np.maximum(0.0, yy2 - yy1 + offset)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def get_detections(pred_boxes, scores, num_classes, nms_threshold, scale_factor = 3.0):
    """
    Get the detections for model predicted boxes and scores.
    Uses tycho_label_map to get the class-wise thresholds.

    Args:
        pred_boxes (numpy.ndarray): Predicted bounding boxes.
        scores (numpy.ndarray): Scores for each class.
        num_classes (int): Number of classes.
        nms_threshold (float): Threshold for non-maximum suppression.
        scale_factor (float, optional): Scale factor for the boxes. Defaults to 3.0.

    Returns:
        numpy.ndarray: Bounding boxes.
        numpy.ndarray: Detections.

    """
    detections = []
    for class_idx in range(1, num_classes):
        class_params = tycho_label_map[class_idx]

        boxes = pred_boxes[:, 4*class_idx:4*(class_idx+1)]
        boxes = boxes * scale_factor
        s = scores[:, class_idx]
        keep = s > (class_params["threshold"])
        dets = np.hstack((boxes[keep], s[keep, np.newaxis])).astype(np.float32)

        # Apply non-max supression.
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]

        # Add class index as the last element
        dets = np.insert(dets, dets.shape[1], class_idx, axis=1)

        detections.append(dets)

    detections = np.concatenate(detections, axis=0)

    return boxes, detections

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        BOX_COLOR,
        -1,
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes_and_classes):
    """
    Visualizes bounding boxes and classes on an image.

    Args:
        image (numpy.ndarray): The input image.
        bboxes_and_classes (list): A list of bounding boxes and classes.

    Returns:
        numpy.ndarray: The image with bounding boxes and classes visualized.
    """
    img = image.copy()
    for bbox_and_class in bboxes_and_classes:
        x1, y1, x2, y2 = bbox_and_class[:4]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        clas = str(int(bbox_and_class[-1]))
        img = visualize_bbox(img, bbox, clas)
    return img

def process_image(img, infer_request):
    """
    Process an image using the Tycho model.

    Args:
        img (numpy.ndarray): The input image.
        infer_request (InferRequest): The inference request object.

    Returns:
        numpy.ndarray: The annotated image with detected objects.
    """
    # Preprocess image
    h = 360
    w = 640
    img_resized, ratio = resize_and_mean_sub(img, (360, 640))

    # Create tensor from image
    input_tensor = np.expand_dims(img_resized, 0)
    input_iminfo = np.expand_dims(np.array([h, w, 1], dtype=float), 0)
    
    # Run inference
    infer_request.infer({0: input_tensor, 1: input_iminfo})

    # Retrieve inference results 
    rois = infer_request.get_output_tensor(9).data
    boxes = rois[:, 1:5]
    box_deltas = infer_request.get_output_tensor(0).data
    scores = infer_request.get_output_tensor(1).data

    # Calculate predicted boxes
    pred_boxes = bbox_transform_inv(boxes, box_deltas)

    num_classes = scores.shape[1] - 1
    nms_threshold = 0.3
    scale_factor = 1 / ratio  # 360x640 -> 1920x1080

    # Get final detections
    bboxes, detections = get_detections(pred_boxes, scores, num_classes, nms_threshold, scale_factor)

    # Drop the confidence column
    detections_viz = np.delete(detections, [4], axis=1)

    # Annotate and return the image
    img_ann = visualize(img, detections_viz)
    return img_ann

def process_video(input_file, output_dir, infer_request):
    """
    Process a video file by reading each frame at specified FPS, performing inference on the frame,
    and saving the annotated frame to the output directory.

    Args:
        input_file (str): Path to the input video file.
        output_dir (str): Path to the output directory where annotated frames will be saved.
        infer_request: The inference request object used for performing inference on each frame.

    Returns:
        None
    """
    if input_file.endswith(".mp4"):
        video = cv2.VideoCapture(input_file)
        if not video.isOpened():
            print(f"Error: Could not open video file")
            return

        # Get the video's original frames per second (fps)
        original_fps = video.get(cv2.CAP_PROP_FPS)

        # Calculate the frame skip value
        frame_skip = round(original_fps / TARGET_FPS) if TARGET_FPS > 0 else 1

        frame_count = 0

        while True:
            # Read the next frame from the video
            ret, frame = video.read()

            # If the frame was not successfully read, then we have reached the end of the video
            if not ret:
                break

            # If the current frame count is a multiple of the frame skip value, process the frame
            if frame_count % frame_skip == 0:

                print(f'Processing: {input_file} : {str(frame_count)}') 
                img_ann = process_image(frame, infer_request)
                
                # Get the base directory name from the input file
                base_dir = os.path.basename(os.path.dirname(input_file))

                # Use the base name to create the output filename
                filename = f'{base_dir}_frame_{frame_count}.jpg'

                outpath = os.path.join(output_dir, filename)
                print(f'Writing to: {outpath}')
                cv2.imwrite(outpath, img_ann)

                print('\n----------------\n')


            frame_count += 1

        # Release the video file
        video.release()


def process_folder(input_dir, output_dir, infer_request):
    """
    Process a folder of images using an inference request.

    Args:
        input_dir (str): The directory path containing the input images.
        output_dir (str): The directory path to save the processed images.
        infer_request: The inference request object.

    Returns:
        None
    """
    # Read input images
    for filename in os.listdir(input_dir):
        if is_image(filename):
            inpath = os.path.join(input_dir, filename)
            print(f'Processing: {inpath}') 

            # Load BGR image
            img = cv2.imread(inpath)


            img_ann = process_image(img, infer_request)

            outpath = os.path.join(output_dir, filename)
            print(f'Writing to: {outpath}')
            cv2.imwrite(outpath, img_ann)

            print('\n----------------\n')


def main(input, mode, model_name, output_dir):

    print(f"=======================================================")
    print(f"intput_dir = {input}")
    print(f"mode       = {mode}")
    print(f"model_name = {model_name}")
    print(f"output_dir = {output_dir}")
    print(f"=======================================================")

    # Initialize OpenVINO Runtime core
    core = ov.Core()
    # Read a model
    model = core.read_model(model_name)

    ############################################################################################################
    
    num_inputs = len(model.inputs)
    print(f"\n\nThe model has {num_inputs} inputs.")
    for i in range(num_inputs):
        input_name = model.input(i)
        print(f"Input name: {input_name}")
    print(f"\n\n")

    
    ############################################################################################################
    num_outputs = len(model.outputs)
    print(f"\n\nThe model has {num_outputs} outputs.")
    for i in range(num_outputs):
        output_name = model.output(i)
        print(f"Output name: {output_name}")
    print(f"\n\n")
    ############################################################################################################

    print('type of input: ', model.input(1))

    # Inizialize Preprocessing for the model
    ppp = PrePostProcessor(model)
    # Specify model's input layout
    ppp.input(0).model().set_layout(Layout("NCHW"))
    # Specify output results format
    ppp.output(0).tensor().set_element_type(Type.f32)
    ppp.output(1).tensor().set_element_type(Type.f32)
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

    if os.path.isfile(input):
        process_video(input, output_dir, infer_request)
    elif os.path.isdir(input):
        process_folder(input, output_dir, infer_request)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and visualise the bboxes on output image")
    parser.add_argument(
        "-a",
        "--input",
        type=str,
        help="path to video file (.mp4) or directory with images ",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--mode",
        type=str,
        help="name of the mode CPU or HDDL",
        required=True,
    )
    parser.add_argument("-c", "--model_name", type=str, help="Path to the model xml", required=True,)
    parser.add_argument(
        "-d",
        "--output_dir",
        type=str,
        help="Path to the directory to save the outputs",
        required=True,
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):  
        os.makedirs(args.output_dir) 

    main(args.input, args.mode, args.model_name, args.output_dir)


# python /data/data2/raviprasad/scripts/main_tycho_yolo.py --input /data/raviprasad/AN/alerts/all/0da6bab8-4de8-4f99-aeef-ce297850f132/0.mp4 --mode CPU --model_name /data/data2/openvino_models/home/ubuntu/autocam/tycho_v5.0.5/openvinomodel.xml  --output_dir /data/data2/raviprasad/tmp_outputs/
# python /data/data2/raviprasad/scripts/main_tycho_yolo.py --input /data/data2/raviprasad/sample_images/ --mode CPU --model_name /data/data2/openvino_models/home/ubuntu/autocam/tycho_v5.0.5/openvinomodel.xml  --output_dir /data/data2/raviprasad/tmp_outputs/
