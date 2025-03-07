#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace std;

const float SCORE_THRESHOLD = 0.25;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.25;


struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
    vector<float> det;
};

struct Resize
{
    cv::Mat resized_image;
    cv::Mat resized_cropped_image;
    int dw;
    int dh;
};

void detect(vector<float>& values){
    int gridX, gridY;
    int anchor_gridX, anchor_gridY;

    int xy_index[] = {0,1,4,5,8,9,12,13,16,17,20,21,22,23,24,25,26,27,28,29};
    int wh_index[] = {2,3,6,7,10,11,14,15,18,19};

    float anchorX[] = { 1.25,  2.0,  4.125, 1.875,  3.875,  3.6875, 3.625,  4.875,  11.65625};
    float anchorY[] = { 1.625, 3.75, 2.875, 3.8125, 2.8125, 7.4375, 2.8125, 6.1875, 10.18750};
    
    int num_filters[] = {5760,1440,360};
    int filter_size1[] = {48,24,12};
    int filter_size2[] = {40,20,10};

    int stride = 0, level=0;
    vector<float> detection(42, 0);

    for (int c=0; c < values.size(); c += 42) {

        int ci = (int)(c/42);
        if (ci<num_filters[0]) {
            gridX = (ci%(filter_size1[0]*filter_size2[0]))%filter_size2[0];
            gridY = (int)((ci%(filter_size1[0]*filter_size2[0]))/filter_size2[0]);
            anchor_gridX = anchorX[((int)(ci/(filter_size1[0]*filter_size2[0])))]*8;
            anchor_gridY = anchorY[((int)(ci/(filter_size1[0]*filter_size2[0])))]*8;
            stride = 8;
            level = 0;
        }
        else if (ci>=num_filters[0] && ci<num_filters[0]+num_filters[1]){
            gridX = ((ci-num_filters[0])%(filter_size1[1]*filter_size2[1]))%filter_size2[1];
            gridY = (int)(((ci-num_filters[0])%(filter_size1[1]*filter_size2[1]))/filter_size2[1]);
            anchor_gridX = anchorX[(int)((ci-num_filters[0])/(filter_size1[1]*filter_size2[1]))+3]*16;
            anchor_gridY = anchorY[(int)((ci-num_filters[0])/(filter_size1[1]*filter_size2[1]))+3]*16;
            stride = 16;
            level = 1;
        }
        else{
            gridX = ((ci-num_filters[1])%(filter_size1[2]*filter_size2[2]))%filter_size2[2];
            gridY = (int)(((ci-num_filters[1])%(filter_size1[2]*filter_size2[2]))/filter_size2[2]);
            anchor_gridX = anchorX[int(((ci-num_filters[0]-num_filters[1])/(filter_size1[2]*filter_size2[2])))+6]*32;
            anchor_gridY = anchorY[int(((ci-num_filters[0]-num_filters[1])/(filter_size1[2]*filter_size2[2])))+6]*32;
            stride = 32;
            level = 2;
        }

        // FOR XY CO-ORDINATES
        for(int i = 0; i < 20; i+=2){
            if(i < 2){
                detection[i]   = (float)(values[c+xy_index[i]  ]*2-0.5+gridX)*stride;
                detection[i+1] = (float)(values[c+xy_index[i+1]]*2-0.5+gridY)*stride;
            }
            else{
                detection[i]   = (float)(values[c+xy_index[i]  ]*(30/(2*level+1))-(15/(2*level+1))+gridX)*stride;
                detection[i+1] = (float)(values[c+xy_index[i+1]]*(30/(2*level+1))-(15/(2*level+1))+gridY)*stride;
            }
        }

        // FOR WH values
        for(int i = 0; i < 10; i+=2){
            if(i < 2){
                detection[20+i]   = values[c+wh_index[i]]*2*values[c+wh_index[i]]*2*anchor_gridX;
                detection[20+i+1] = values[c+wh_index[i+1]]*2*values[c+wh_index[i+1]]*2*anchor_gridY;
            }
            else{
                detection[20+i]   = values[c+wh_index[i]]*values[c+wh_index[i]]*anchor_gridX;
                detection[20+i+1] = values[c+wh_index[i+1]]*values[c+wh_index[i+1]]*anchor_gridY;
            }
        }

        for(int i = 0; i < 30; i++){
            values[c+i] = detection[i];
        }
    }
}

int main(int argc, char** argv){

    float a;
    vector<float> values;

    std::ifstream fin(argv[1], std::ios::binary);
    while (fin.read(reinterpret_cast<char*>(&a), sizeof(float))){
        values.push_back(a);
    }

    detect(values);
    cout << "[DEBUG] " << values.size() << endl;
    float *detections = values.data();

    int output_shape[3]={1,7560,42};

    Resize res;
    res.resized_cropped_image = cv::imread(argv[2]);

    // Step 8. Postprocessing including NMS 
    std::vector<vector<float>> box_floats;
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    int flag = 1;
    for (int i = 0; i < output_shape[1]; i++){
        float *detection = &detections[i * output_shape[2]];
        float confidence = detection[30];
        //cout << "[DEBUG] conf " << confidence  << detection[20] << endl;

        //if (confidence >= CONFIDENCE_THRESHOLD){
            float person_class_score = detection[30];
            //if (person_class_score > SCORE_THRESHOLD){

                confidences.push_back(confidence);
                class_ids.push_back(30);

                ////////////////////////////////////////////////// pbox
                float x = detection[0];
                float y = detection[1];
                float w = detection[20];
                float h = detection[21];

                float xmin = x - (w / 2);
                float ymin = y - (h / 2);

                std::vector<float> temp{detection, detection+42};
                box_floats.push_back(temp);
                boxes.push_back(cv::Rect(xmin, ymin, w, h));

            //}
        //}
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<Detection> output;
    int idx;
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        idx = nms_result[i];
        cout << " NMS_RESULT_INDEX : " << idx << endl;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.det = box_floats[idx];
        cout << " NMS_RESULT : " << box_floats[idx][0] << " " << box_floats[idx][1] << " " << box_floats[idx][20] << " " << box_floats[idx][21] << " " << endl;
        output.push_back(result);
    }

    cout << " NMS_RESULT_SIZE : " << nms_result.size() << endl;
    ofstream file1( "result.txt",ios::out | ios::app );
    file1 << argv[1] << "\n";
    // Step 9. Print results and save Figure with detections

    /*
    if(nms_result.size()==0){
        file1 << "1299921" << "\n";
    }
    else{
        for (int i = 0; i < output.size(); i++)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            auto confidence = detection.confidence;
            auto det_floats = detection.det;

            float rx = 1.0;
            float ry = 1.0;
            box.x = rx * box.x; // AS THIS IS CALCULATED FOR CROPPED IMAGE WE NEED TO SHIFT BY 320 TO PUT BOXES WHERE THEY BELONG
            box.y = ry * box.y;
            box.width = rx * box.width;
            box.height = ry * box.height;
            cout << "network outputs------------------------------------------" << endl;
            cout << "Bbox" << i + 1 << ": Class: " << classId << " "
             << "Confidence: " << confidence << " Scaled coords: [ "
             << "cx: " << det_floats[0] << ", "
             << "cy: " << det_floats[1] << ", "
             << "w: " << det_floats[2] << ", "
             << "h: " << det_floats[3] << " ]" << endl;
        
            for(int i = 0; i < 42; i++){
                file1 << det_floats[i];
                if(i<41) file1 << " ";
                else file1 << "\n";
            }
        }
    }
    file1.close();
    */

    //cv::imwrite("./detection_cpp.png", res.resized_cropped_image);
    
    for (int i = 0; i < output.size(); i++)
    {
        float xmin, ymin, xmax, ymax;

        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;
        auto det = detection.det;
        float rx = 1.0;
        float ry = 1.0;
        box.x = rx * box.x; // AS THIS IS CALCULATED FOR CROPPED IMAGE WE NEED TO SHIFT BY 320 TO PUT BOXES WHERE THEY BELONG
        box.y = ry * box.y;
        box.width = rx * box.width;
        box.height = ry * box.height;
        cout << "UN-NORMALIZED------------------------------------------" << endl;
        cout << "Bbox" << i + 1 << ": Class: " << classId << " "
             << "Confidence: " << confidence << " Scaled coords: [ "
             << "cx: " << (float)(box.x + (box.width / 2)) << ", "
             << "cy: " << (float)(box.y + (box.height / 2)) << ", "
             << "w: " << (float)box.width << ", "
             << "h: " << (float)box.height << " ]" << endl;
        cout << "Bbox" << i + 1 << ": Class: " << classId << " "
             << "Confidence: " << confidence << " Scaled coords: [ "
             << "cx: " << box_floats[idx][0] << ", "
             << "cy: " << box_floats[idx][1] << ", "
             << "w: " << box_floats[idx][20] << ", "
             << "h: " << box_floats[idx][21] << " ]" << endl;
        xmax = box.x + box.width;
        ymax = box.y + box.height;
        cv::rectangle(res.resized_cropped_image, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 1);
        cv::rectangle(res.resized_cropped_image, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(res.resized_cropped_image, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // EYE1
        xmin = det[4]-(det[24]/2);
        ymin = det[5]-(det[25]/2);
        xmax = xmin + det[24];
        ymax = ymin + det[25];
        cv::rectangle(res.resized_cropped_image, cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(255, 0, 0), 1);
        // EYE2
        xmin = det[6]-(det[26]/2);
        ymin = det[7]-(det[27]/2);
        xmax = xmin + det[26];
        ymax = ymin + det[27];
        cv::rectangle(res.resized_cropped_image, cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(255, 0, 21), 1);
        //FACE
        xmin = det[8]-(det[28]/2);
        ymin = det[9]-(det[29]/2);
        xmax = xmin + det[28];
        ymax = ymin + det[29];
        cv::rectangle(res.resized_cropped_image, cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(0, 0, 255), 1);

    }
    cv::imwrite("./detection_cpp.png", res.resized_cropped_image);
    cv::imwrite(string(argv[3])+"/output.png", res.resized_cropped_image);
    
    return 0;
}