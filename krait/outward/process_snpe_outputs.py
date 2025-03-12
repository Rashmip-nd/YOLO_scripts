# Command for PTQ
# python3 process_snpe_outputs.py -i <snpe_inference_output_folder> -l <imageset> -o <folder_to_dump_renamed_output_images> -q ptq

# Command for QAT
# python3 process_snpe_outputs -i <snpe_inference_output_folder> -l <imageset> -o <folder_to_dump_renamed_output_images> -q qat -op1 <output1>.raw -op2 <output2>.raw -op3 <output3>.raw

import os
import re
import time
import argparse
import numpy as np
from shutil import copy

parser = argparse.ArgumentParser(description='Rename output raw folder as per the annotation of dataset')
parser.add_argument('-i', '--input_folder', help='File containing output raw images', required=True)
parser.add_argument('-l', '--imageset', help='File containing image names without complete path and extension', required=True)
parser.add_argument('-o', '--output_folder', help='Folder to dump the raw output file', required=True)
parser.add_argument('-q', '--quant_type', help='ptq/qat type of quantization', required=True)
parser.add_argument('-op1', '--output1', help='first output layer name if quant_type is qat', required=False)
parser.add_argument('-op2', '--output2', help='second output layer name if quant_type is qat', required=False)
parser.add_argument('-op3', '--output3', help='third output layer name if quant_type is qat', required=False)
args = parser.parse_args()

input_folder = os.path.abspath(args.input_folder)
imageset = os.path.abspath(args.imageset)
output_folder = os.path.abspath(args.output_folder)
quant_type = str(args.quant_type)
output1 = str(args.output1)
output2 = str(args.output2)
output3 = str(args.output3)

nc = 14 # num classes

no = nc + 5  # num outputs (nc + bboxes + objectness)
na = 3  # num anchors
nl = 3  # num det layers
fh = 48  # feature height
fw = 80  # feature width

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

prediction_shape = (1, 15120, 19)

# output1 = 'Conv_output_00.raw'
# output2 = 'Conv_output_01.raw'
# output3 = 'Conv_output_02.raw'

# output1 = '1047.raw'
# output2 = '1066.raw'
# output3 = '1085.raw'

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def additional_postprocessing_snpe(pred1, pred2, pred3):
	#AIMET QAT postprocessing
	pred1 = pred1.reshape(shape1_1)
	pred2 = pred2.reshape(shape2_1)
	pred3 = pred3.reshape(shape3_1)

	pred1= np.transpose(pred1, (0, 3, 1, 2))
	pred2= np.transpose(pred2, (0, 3, 1, 2))
	pred3= np.transpose(pred3, (0, 3, 1, 2))

	pred1 = pred1.reshape(shape1_3)
	pred2 = pred2.reshape(shape2_3)
	pred3 = pred3.reshape(shape3_3)

	pred1= np.transpose(pred1, (0, 2, 3, 1))
	pred2= np.transpose(pred2, (0, 2, 3, 1))
	pred3= np.transpose(pred3, (0, 2, 3, 1))
		
	pred1 = pred1.reshape(shape1)
	pred2 = pred2.reshape(shape2)
	pred3 = pred3.reshape(shape3)
	
	concat_pred = np.concatenate((pred1, pred2, pred3), axis=1)
	
	final_pred = sigmoid(concat_pred)
	
	return final_pred


def load_images_from_folder(input_folder, imageset, output_folder):
	ip_file_object = open(imageset, "r")
	ip_read_lines = ip_file_object.readlines()

	image_name_list = []

	i = 0

	for ip_img in ip_read_lines:
		ip_img = ip_img.rstrip()

		image_name_list.append(ip_img)

	#(image_name_list)

	ip_list = os.listdir(input_folder)
	#print("\n\ndir list : ", ip_list)
	ip_list = sorted(ip_list, key=lambda x: int(x.partition('_')[2]) if x.partition('_')[2].isdigit() else float('inf'))
	#print("\n\nsorted dir list : ", ip_list)

	for op_folder in ip_list:
		if "log" in op_folder:
			continue

		print("Reading folder : ", op_folder)
		result_path = os.path.join(input_folder, op_folder)
		if 'Result' in result_path:
			raw_image_name = image_name_list[i] + '.raw'
			i+=1

			if quant_type == 'qat':
				predpath1 = os.path.join(result_path, output1)
				predpath2 = os.path.join(result_path, output2)
				predpath3 = os.path.join(result_path, output3)
				print(predpath1)
				print(predpath2)
				print(predpath3)
				
				pred1 = np.fromfile(predpath1, dtype=np.float32)
				pred2 = np.fromfile(predpath2, dtype=np.float32)
				pred3 = np.fromfile(predpath3, dtype=np.float32)
				
				prediction_np = additional_postprocessing_snpe(pred1, pred2, pred3)
				print(prediction_np.shape)

				prediction_np = prediction_np.reshape(prediction_shape[0]*prediction_shape[1]*prediction_shape[2])

				prediction_np.tofile(os.path.join(output_folder,raw_image_name))
			elif quant_type == 'ptq':
				for raw_file in os.listdir(result_path):
					print("raw_file : ", os.path.join(result_path,raw_file))
					copy(os.path.join(result_path,raw_file), os.path.join(output_folder,raw_image_name))
					break

	print("Completed conversion")
	ip_file_object.close()

load_images_from_folder(input_folder, imageset, output_folder)