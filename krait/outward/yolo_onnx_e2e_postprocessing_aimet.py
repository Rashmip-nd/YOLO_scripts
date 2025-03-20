'''
usage: yolo_onnx_e2e.py [-h] -d DIRPATH -i IMGNAME [-m MODELNAME] [-p PREDNAME]
Run inference and visualise the bboxes on output image
optional arguments:
  -h, --help            show this help message and exit
  -d DIRPATH, --dirpath DIRPATH
                        directory with image and model
  -i IMGNAME, --imgname IMGNAME
                        name of the jpg image (without extension)
  -m MODELNAME, --modelname MODELNAME
                        name of the onnx model (without extension)
  -p PREDNAME, --predname PREDNAME
                        name of the prediction numpy file (with extension if json file else without extension for others) 
'''

import os
import time
import argparse
import cv2
import torch
import numpy as np
import onnxruntime
import torchvision
import json

#YOLO_MODEL = 'standard'  #standard / nd
YOLO_MODEL = 'nd'

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

if YOLO_MODEL == 'standard':
    input_img_size = [640, 640]
else:
    input_img_size = [384, 640]

def aspect_aware_resize(image_path, img_size=input_img_size):
    im = cv2.imread(image_path).astype(np.float32)
    assert im is not None, f'Image Not Found {image_path}'
    h0, w0 = im.shape[:2]
    r = max(img_size) / max(h0, w0)  # ratio
    interp = cv2.INTER_LINEAR # if (r > 1) else cv2.INTER_AREA
    im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
    print(im.shape)
    return im

def resize_letterbox(image_path, img_size=input_img_size):
    """Perform aspect preserving resize and add letterbox padding"""
    im = aspect_aware_resize(image_path, img_size=img_size)
    img, ratio, pad = letterbox(im, img_size, auto=False, scaleup=False)
    return img

def preprocess(image_path, img_size=input_img_size):
    """Preprocess image for yolo (resize, pad, channel transpose)"""
    img = resize_letterbox(image_path, img_size=img_size)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img) / 255.0
    return img[None, :]

def postprocess(image_data, img_size=input_img_size):
    image_data_len = len(image_data)
    print(image_data.shape)
    
    for c in range(0, image_data_len, 19):

        if c < 4:
            print("[debug] ",image_data[c], image_data[c+1])
        
        cx = image_data[c]
        cy = image_data[c+1]
        w = image_data[c+2]
        h = image_data[c+3]

        gridX = 0
        gridY = 0
        anchor_gridX = 0
        anchor_gridY = 0

        #anchorX = [10,16,33,30,62,59,116,156,373]
        #anchorY = [13,30,23,61,45,119,90,198,326]
        #anchorX = [ 6.1758, 7.8828, 5.8359, 11.7344, 16.6094, 42.7813, 24.3907, 44.9686, 80.4998]
        #anchorY = [ 4.6914, 7.4375, 12.7969, 10.5938, 16.0626, 10.3203, 25.8595, 43.4064, 84.9376]
        anchorX = [7.0, 5.0, 10.0, 9.0, 16.0, 40.0, 24.0, 43.0, 71.0]
        anchorY = [5.0, 11.0, 9.0, 20.0, 17.0, 9.0, 29.0, 46.0, 96.0]

        if YOLO_MODEL == 'standard':
            num_filters = [19200,4800,1200]
            filter_size1 = [80,40,20]
            filter_size2 = [80,40,20]
        else:
            num_filters = [11520,2880,720]
            filter_size1 = [48,24,12]
            filter_size2 = [80,40,20]

        stride = 0
        ci = int(((c+4)/19));
        
        if ci<num_filters[0]:
            gridX = (ci%(filter_size1[0]*filter_size2[0]))%filter_size2[0]
            gridY = (int)((ci%(filter_size1[0]*filter_size2[0]))/filter_size2[0])
            anchor_gridX = anchorX[((int)(ci/(filter_size1[0]*filter_size2[0])))]
            anchor_gridY = anchorY[((int)(ci/(filter_size1[0]*filter_size2[0])))]
            stride = 8;
        elif (ci>=num_filters[0] and ci<num_filters[0]+num_filters[1]):
            gridX = ((ci-num_filters[0])%(filter_size1[1]*filter_size2[1]))%filter_size2[1]
            gridY = (int)(((ci-num_filters[0])%(filter_size1[1]*filter_size2[1]))/filter_size2[1])
            anchor_gridX = anchorX[(int)((ci-num_filters[0])/(filter_size1[1]*filter_size2[1]))+3]
            anchor_gridY = anchorY[(int)((ci-num_filters[0])/(filter_size1[1]*filter_size2[1]))+3]
            stride = 16
        else:
            gridX = ((ci-num_filters[1])%(filter_size1[2]*filter_size2[2]))%filter_size2[2]
            gridY = (int)(((ci-num_filters[1])%(filter_size1[2]*filter_size2[2]))/filter_size2[2])
            anchor_gridX = anchorX[int(((ci-num_filters[0]-num_filters[1])/(filter_size1[2]*filter_size2[2])))+6]
            anchor_gridY = anchorY[int(((ci-num_filters[0]-num_filters[1])/(filter_size1[2]*filter_size2[2])))+6]
            stride = 32
        
        cx = float((cx*2-0.5+gridX)*stride);
        cy = float((cy*2-0.5+gridY)*stride);
        w = w*2*w*2*anchor_gridX;
        h = h*2*h*2*anchor_gridY;

        #print("cx, cy, w, h")
        #print(cx, cy, w, h)
        image_data[c]   = cx
        image_data[c+1] = cy
        image_data[c+2] = w
        image_data[c+3] = h
    
    return image_data

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)

def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    print("prediction.shape : ", prediction.shape)

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    print("xc : ", xc)

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            print("if labels")
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]: 
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    return output

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
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
    #print('bboxes_and_classes ', bboxes_and_classes)
    img = image.copy()
    for bbox_and_class in bboxes_and_classes:
        x1, y1, x2, y2 = bbox_and_class[:4]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        clas = str(int(bbox_and_class[-1]))
        img = visualize_bbox(img, bbox, clas)
    return img

if YOLO_MODEL == 'standard':
    shape1 = (1, 19200, 19)
    shape2 = (1, 4800, 19)
    shape3 = (1, 1200, 19)
else:
    shape1 = (1, 11520, 19)
    shape2 = (1, 2880, 19)
    shape3 = (1, 720, 19)

    shape1_1 = (1, 48, 80, 57)
    shape2_1 = (1, 24, 40, 57)
    shape3_1 = (1, 12, 20, 57)

    shape1_2 = (1, 57, 48, 80)
    shape2_2 = (1, 57, 24, 40)
    shape3_2 = (1, 57, 12, 20)

    shape1_3 = (3, 19, 48, 80)
    shape2_3 = (3, 19, 24, 40)
    shape3_3 = (3, 19, 12, 20)

    shape1_4 = (3, 48, 80, 19)
    shape2_4 = (3, 24, 40, 19)
    shape3_4 = (3, 12, 20, 19)

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def additional_postprocessing_snpe(pred1, pred2, pred3):
    #AIMET QAT postprocessing
    pred1 = pred1.reshape(shape1_1)
    pred2 = pred2.reshape(shape2_1)
    pred3 = pred3.reshape(shape3_1)
    print(pred1.shape, " ", pred2.shape, " ", pred3.shape)

    pred1= np.transpose(pred1, (0, 3, 1, 2))
    pred2= np.transpose(pred2, (0, 3, 1, 2))
    pred3= np.transpose(pred3, (0, 3, 1, 2))
    print(pred1.shape, " ", pred2.shape, " ", pred3.shape)

    pred1 = pred1.reshape(shape1_3)
    pred2 = pred2.reshape(shape2_3)
    pred3 = pred3.reshape(shape3_3)
    print(pred1.shape, " ", pred2.shape, " ", pred3.shape)

    pred1= np.transpose(pred1, (0, 2, 3, 1))
    pred2= np.transpose(pred2, (0, 2, 3, 1))
    pred3= np.transpose(pred3, (0, 2, 3, 1))
    print(pred1.shape, " ", pred2.shape, " ", pred3.shape)

    pred1 = pred1.reshape(shape1)
    pred2 = pred2.reshape(shape2)
    pred3 = pred3.reshape(shape3)
    print(pred1.shape, " ", pred2.shape, " ", pred3.shape)

    concat_pred = np.concatenate((pred1, pred2, pred3), axis=1)

    final_pred = sigmoid(concat_pred)

    return final_pred

def additional_postprocessing_onnx(pred1, pred2, pred3):
    pred1 = pred1.reshape(shape1)
    pred2 = pred2.reshape(shape2)
    pred3 = pred3.reshape(shape3)
    print(pred1.shape, " ", pred2.shape, " ", pred3.shape)

    concat_pred = np.concatenate((pred1, pred2, pred3), axis=1)

    final_pred = sigmoid(concat_pred)

    return final_pred

def run(dirpath = "/data/raviprasad/scripts",
        imgname = None,
        modelname = "yolo_s103_exp2_simp_noOpt_384x640",
        predname1 = None,
        predname2 = None,
        predname3 = None
        ):
    
    if not (modelname or predname1):
        raise Exception("Either modelname or prediction numpy file name must be provided! Exiting...")
    
    imgpath = os.path.join(dirpath, f"{imgname}.jpg")
    after_detect = os.path.join(dirpath, f"{predname1}.bin")

    if not predname1:
        # load image, resize and add padding
        print("$$$$$$$$$$$$$$ ONNX_EXECUTION $$$$$$$$$$$$$")
        img = preprocess(imgpath)

        # load onnx model
        modelpath = os.path.join(dirpath, f"{modelname}.onnx")
        session = onnxruntime.InferenceSession(modelpath)
        ort_inputs = {session.get_inputs()[0].name: img}

        # run inference 
        prediction = session.run(None, ort_inputs)
        print("ONNX predictions 0 : ", prediction[0].shape)
        print("ONNX predictions 1 : ", prediction[1].shape)
        print("ONNX predictions 2 : ", prediction[2].shape)
        prediction_np = additional_postprocessing_onnx(prediction[0], prediction[1], prediction[2])

        #prediction_np = prediction[0]
            # predpath = os.path.join(dirpath, f"{imgname}_onnx.raw")
            # np.save(predpath, prediction_np[0])
            # prediction_np.tofile(predpath)
            # prediction_shape = prediction_np.shape
            # print(prediction_shape)
        # load npy
            # prediction_np = np.load(predpath)
            # prediction_np = np.fromfile(predpath, dtype=np.float32)
            # np.savetxt('fp32_array.txt', prediction_np)
            # print('loaded raw shape: ', prediction_np.shape)

    else:
        predpath1 = os.path.join(dirpath, f"{predname1}.raw")
        predpath2 = os.path.join(dirpath, f"{predname2}.raw")
        predpath3 = os.path.join(dirpath, f"{predname3}.raw")

        # load npy
        #prediction_np1 = np.load(predpath1)
        prediction_np1 = np.fromfile(predpath1, dtype=np.float32)
        #prediction_np2 = np.load(predpath2)
        prediction_np2 = np.fromfile(predpath2, dtype=np.float32)
        #prediction_np3 = np.load(predpath3)
        prediction_np3 = np.fromfile(predpath3, dtype=np.float32)
        #np.savetxt('fp32_array.txt', prediction_np)
        print('loaded raw shape1 : ', prediction_np1.shape)
        print('loaded raw shape2 : ', prediction_np2.shape)
        print('loaded raw shape3 : ', prediction_np3.shape)

        prediction_np = additional_postprocessing_snpe(prediction_np1, prediction_np2, prediction_np3)
        
    if YOLO_MODEL == 'standard':
        prediction_shape = (1, 25200, 19)
    else:
        prediction_shape = (1, 15120, 19)
    print(prediction_shape)
    
    prediction_np = prediction_np.reshape(prediction_shape[0]*prediction_shape[1]*prediction_shape[2])

    # run postprocessing
    start_time = time.perf_counter() #time.clock()
    prediction_np = postprocess(prediction_np)
    print("TIME for DETECT ------------------------ ", time.perf_counter() - start_time)
    
    prediction_np.tofile(after_detect)

    prediction_np = prediction_np.reshape(prediction_shape)
    prediction_tensor = torch.from_numpy(prediction_np)
    
    # run non max suppression
    output = non_max_suppression(prediction_tensor)
    print('output shape : ', output[0].shape)

    # draw annotations on image
    img = resize_letterbox(imgpath)
    img_ann = visualize(img, output[0])

    # save the result image file
    outpath = os.path.join(dirpath, f"{imgname}_{YOLO_MODEL}_{args.modelname}.jpg")
    print('Saving result to: ', outpath)
    ret = cv2.imwrite(outpath, img_ann)


parser = argparse.ArgumentParser(description="Run inference and visualise the bboxes on output image")
parser.add_argument("-d", "--dirpath", type=str, help="directory with image and model", required=True)
parser.add_argument("-i", "--imgname", type=str, help="name of the jpg image (without extension)", required=True)
parser.add_argument("-m", "--modelname", type=str, help="name of the onnx model (without extension)")
parser.add_argument("-p1", "--predname1", type=str, help="name of the prediction numpy file (without extension)")
parser.add_argument("-p2", "--predname2", type=str, help="name of the prediction numpy file (without extension)")
parser.add_argument("-p3", "--predname3", type=str, help="name of the prediction numpy file (without extension)")
args = parser.parse_args()

if __name__ == "__main__":
    run(args.dirpath, args.imgname, args.modelname, args.predname1, args.predname2, args.predname3)

# python /data/raviprasad/scripts/yolo_onnx_e2e.py -d "/data/raviprasad/scripts" -i "a6c7af98-8a9e-465a-b806-a8059d07fe51_000001" -m "yolo_s103_exp2_simp_noOpt_384x640"
# or
# python /data/raviprasad/scripts/yolo_onnx_e2e.py -d "/data/raviprasad/scripts" -i "a6c7af98-8a9e-465a-b806-a8059d07fe51_000001" -p "a6c7af98-8a9e-465a-b806-a8059d07fe51_000001