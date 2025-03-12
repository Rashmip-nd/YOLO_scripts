#python3 outward_preprocessing.py -t <file_with_jpg_images_complete_path> -i <folder_with_jpg_images> -o <folder_path_to_dump_raw_images> -l <input_list>
import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Convert jpeg image to float values')
parser.add_argument('-t', '--input_image_text_file', help='Text file containing input image filename', required=True)
parser.add_argument('-i', '--input_folder', help='File containing input images', required=True)
parser.add_argument('-o', '--output_folder', help='Output file to dump raw images', required=True)
parser.add_argument('-l', '--input_list', help='File name for creating raw image input list', required=True)
args = parser.parse_args()

input_image_text_file =os.path.abspath(args.input_image_text_file)
input_folder = os.path.abspath(args.input_folder)
output_folder = os.path.abspath(args.output_folder)
input_list = os.path.abspath(args.input_list)

#if not os.path.isfile(input_folder):
#    raise RuntimeError('input_folder %s does not exist' % input_folder)

#if not os.path.isfile(output_folder):
#    raise RuntimeError('output_folder %s does not exist' % output_folder)

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

def cv2_load(image_path, j, img_size=640):
    im = cv2.imread(image_path).astype(np.float32)

    print("Convert BGR to RGB")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    assert im is not None, f'Image Not Found {image_path}'
    h0, w0 = im.shape[:2]
    r = img_size / max(h0, w0)  # ratio
    interp = cv2.INTER_LINEAR # if (r > 1) else cv2.INTER_AREA
    im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
    print(im.shape)
    img, ratio, pad = letterbox(im, [384, 640], auto=False, scaleup=False)
    print("img shape :  , ratio  :  , pad :  ", img.shape, ratio, pad)
    #img = img.transpose((2, 0, 1))[::-1]
    print("image shape after transpose : ", img.shape)

    img = np.ascontiguousarray(img) / 255.0
    print("image shape after ascontiguousarray : ", img.shape)
    return img[None, :]

def load_and_convert_images(input_image_text_file, input_folder, output_folder, input_list):
	i = 0

	file_object = open(input_list, 'w')
	ip_file_object = open(input_image_text_file, "r")
	ip_read_lines = ip_file_object.readlines()

	input_filename_with_ext = os.path.basename(input_image_text_file)
	input_filename = os.path.splitext(input_filename_with_ext)

	for ip_img in ip_read_lines:
		ip_img = ip_img.rstrip()

		resized_raw_image = ip_img + ".raw"

		# in the training nd dataset images are in jpg format and extension is missing in the text file image list so adding it here
		ip_img = ip_img + ".jpg"

		image = cv2_load(os.path.join(input_folder,ip_img), i)
		#image = cv2.imread(os.path.join(input_folder,ip_img))
		#image = cv2.resize(image, (384,640))
		image = image.astype(np.float32)
		np_array = np.asarray(image)
		raw_image_file = os.path.join(output_folder,resized_raw_image)
		np_array.tofile(raw_image_file)
		
		print("Converted " , ip_img, " to raw file ", resized_raw_image)

		file_object.writelines(raw_image_file + '\n')

		i+=1 

	print("Completed conversion")
	file_object.close()
	ip_file_object.close()

load_and_convert_images(input_image_text_file, input_folder, output_folder, input_list)
