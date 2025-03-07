import os
import cv2
import argparse
import numpy as np
from polygraphy.json import save_json

parser = argparse.ArgumentParser(description='Preprocess jpg image for TRT')
parser.add_argument('-i', '--input_images', help='File list containing raw image/s names with full path', required=True)
parser.add_argument('-l', '--json_np_filename', help='Json/Numpy file of preprocessed images', required=False)
args = parser.parse_args()

input_images = os.path.abspath(args.input_images)
json_np_filename = os.path.abspath(args.json_np_filename)

width = 384
height = 640
channels = 3
batch = 1

def read_raw_image_file(file_path, width, height, channels):
     # Calculate the total number of bytes needed
    total_size = width * height * channels * np.dtype(np.float32).itemsize
    print('total_size : ', total_size)
    
    # Read the raw image data
    with open(file_path, 'rb') as f:
        raw_data = f.read(total_size)

    # Convert the raw data to a NumPy array with float32 data type
    image_array = np.frombuffer(raw_data, dtype=np.float32)

    # Reshape the array to the appropriate dimensions
    image_array = image_array.reshape((batch, channels, width, height))

    return image_array

def load_data(input_image, op_img):
    print("image shape while loading data to json : ", op_img.shape)
    yield {
        #str(os.path.basename(input_image)): op_img
        'images': op_img
    }

def load_and_convert_images(input_image_text_file):
    ip_file_object = open(input_image_text_file, "r")
    ip_read_lines = ip_file_object.readlines()

    input_filename_with_ext = os.path.basename(input_image_text_file)
    input_filename = os.path.splitext(input_filename_with_ext)

    image_name = 'images'
    input_data = []

    for ip_img in ip_read_lines:
        ip_img = ip_img.rstrip()

        raw_imgname = os.path.basename(ip_img)
        raw_imgname = os.path.splitext(raw_imgname)[0]
        print("ip_img basename : ", raw_imgname)

        image = read_raw_image_file(os.path.join(ip_img), width, height, channels)
        image = image.astype(np.float32)
        np_array = np.asarray(image)

        input_data.append({image_name: np_array})

    save_json(input_data, json_np_filename, description="custom input data")
    print("Wrote ", input_images, " to json file")

load_and_convert_images(input_images)