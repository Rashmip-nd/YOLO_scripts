import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='trtexec run inference on multiple images')
parser.add_argument('-i', '--input_image_list', help='File containing list of preprocessed raw images', required=True)
parser.add_argument('-m', '--trt_model', help='TRT model for inference', required=True)
parser.add_argument('-o', '--output_folder', help='Output folder to write the trt inference output', required=True)
parser.add_argument('-f', '--bit_format', help='fp32/fp16/int8', required=True)

args = parser.parse_args()
input_image_list = os.path.abspath(args.input_image_list)
trt_model = os.path.abspath(args.trt_model)
output_folder = os.path.abspath(args.output_folder)
bit_format = args.bit_format

count = 0

def read_input_list_and_process(input_image_list, trt_model, output_folder):
	ip_read_raw_image_file = open(input_image_list, 'r')

	for raw_img in ip_read_raw_image_file:
		if count > 600:
			raw_img = raw_img.rstrip()

			raw_imgname = os.path.basename(raw_img)
			raw_imgname = os.path.splitext(raw_imgname)[0]
			#print("ip_img basename : ", raw_imgname)

			if bit_format == 'fp32':
				command = "/usr/src/tensorrt/bin/trtexec --loadEngine=" + trt_model + " --exportOutput=" + output_folder + "/" + raw_imgname + ".json" + " --loadInputs='images:" + raw_img + "'"
			else:
				command = "/usr/src/tensorrt/bin/trtexec --loadEngine=" + trt_model + " --exportOutput=" + output_folder + "/" + raw_imgname + ".json" + " --loadInputs='images:" + raw_img + "'" + " --" + bit_format

			print("command : ", command, "\n\n")
			run_trtexec(command)
		count += 1


def run_trtexec(command):
    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
        
        # Get the standard output (stdout) and standard error (stderr)
        print("Output:\n", result.stdout)
        print("Error (if any):\n", result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")

# Example: Running a sample trtexec command to check TensorRT version
#command = "trtexec --version"
#run_trtexec(command)

read_input_list_and_process(input_image_list, trt_model, output_folder)