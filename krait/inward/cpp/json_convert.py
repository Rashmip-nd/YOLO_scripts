import json
import os
import argparse

def SetupArguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', default='')
    parser.add_argument('--output_file', default = '')

    args = parser.parse_args() 
    return args

args = SetupArguments()

input_file = args.input_file
output_file = args.output_file

dict1 = {}

with open(input_file) as f:
    lines = f.readlines()
    list1=[]
    filename=""

    for line in lines:
        if "/" in line:
            if len(filename) !=0:
                dict1[filename]=list1
                list1=[]
            
            filename = line.rstrip()
            filename = filename.split('/')[-1]
            
        else:
            number = line.rstrip()   
            number = number.replace(' ',',')
            number = [float(i) for i in number.split(',')]
            if len(number) == 1:
                pass
            else:
                list1.append(number)
    dict1[filename]=list1
    
with open(output_file, 'w') as outfile:
    json.dump(dict1, outfile)