# YOLO_scripts
All scripts related to YOLO work will be maintained in here.

Both inward and outward have cpp folders with DETECT layer implementations.

## KRAIT

Inward
    CPP performs both detect(cpp) + NMS(cpp) = output with annotations

Outward
    There are 2 ways you can do post-processing
        1. use detect(cpp) + yolo_onnx_e2e_nms(python) 
        2. yolo_onnx_e2e_postprocessing (both detect and nms in python)

How to compile cpp folders:
Go to cpp folder
```
mkdir build
cd build
cmake -S ../ -O ./
```

## MOWGLI
Scripts related to MOWGLI product line are kept in mowgli folder.

INWARD :
```
python3 main_pose_yolo.py --input_dir <input-image-dir> --mode <CPU,HDDL> --model_name <model.xml path, same location must have model.bin> --output_dir <output-image-dir>
```
OUTWARD :
```
python3 main_tycho_yolo.py --input_dir <input-image-dir> --mode <CPU,HDDL> --model_name <model.xml path, same location must have model.bin> --output_dir <output-image-dir>
```
    
    TODO : Provide model input shape as parameter.
