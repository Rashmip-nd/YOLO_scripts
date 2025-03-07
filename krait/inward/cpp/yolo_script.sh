#!/bin/bash
echo "First arg: $1"

i=1;
for f in $1/*.jpg
do 
    echo $f
    ./main $f
    echo $i
    if [[ "$i" == '2000' ]]
    then
        break
    fi
    i=$((i+1));
done

python3 json_convert.py --input_file result.txt --output_file /app/analytics/yolo/yolo-work/20230704/openvino_CPU5.json