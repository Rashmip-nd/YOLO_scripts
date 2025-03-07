#include <bits/stdc++.h>
using namespace std;
//#define float double

int main(int argc, char** argv){

    float a;
    vector<float> values;

    std::ifstream fin(argv[1], std::ios::binary);
    while (fin.read(reinterpret_cast<char*>(&a), sizeof(float))){
        values.push_back(a);
    }
    cout << "[DEBUG] " << values.size() << endl;
    
    int gridX, gridY;
    int anchor_gridX, anchor_gridY;
    
    float anchorX[] = { 6.1758, 7.8828, 5.8359, 11.7344, 16.6094, 42.7813, 24.3907, 44.9686, 80.4998};
    float anchorY[] = { 4.6914, 7.4375, 12.7969, 10.5938, 16.0626, 10.3203, 25.8595, 43.4064, 84.9376};
    
    int num_filters[] = {11520,2880,720};
    int filter_size1[] = {48,24,12};
    int filter_size2[] = {80,40,20};

    int stride = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int c=0; c < values.size(); c += 17) {

        float cx = values[c];
        float cy = values[c+1];
        float w = values[c+2];
        float h = values[c+3];


        int ci = (int)(c/17);
        if (ci<num_filters[0]) {
            gridX = (ci%(filter_size1[0]*filter_size2[0]))%filter_size2[0];
            gridY = (int)((ci%(filter_size1[0]*filter_size2[0]))/filter_size2[0]);
            anchor_gridX = anchorX[((int)(ci/(filter_size1[0]*filter_size2[0])))];
            anchor_gridY = anchorY[((int)(ci/(filter_size1[0]*filter_size2[0])))];
            stride = 8;
        }
        else if (ci>=num_filters[0] && ci<num_filters[0]+num_filters[1]){
            gridX = ((ci-num_filters[0])%(filter_size1[1]*filter_size2[1]))%filter_size2[1];
            gridY = (int)(((ci-num_filters[0])%(filter_size1[1]*filter_size2[1]))/filter_size2[1]);
            anchor_gridX = anchorX[(int)((ci-num_filters[0])/(filter_size1[1]*filter_size2[1]))+3];
            anchor_gridY = anchorY[(int)((ci-num_filters[0])/(filter_size1[1]*filter_size2[1]))+3];
            stride = 16;
        }
        else{
            gridX = ((ci-num_filters[1])%(filter_size1[2]*filter_size2[2]))%filter_size2[2];
            gridY = (int)(((ci-num_filters[1])%(filter_size1[2]*filter_size2[2]))/filter_size2[2]);
            anchor_gridX = anchorX[int(((ci-num_filters[0]-num_filters[1])/(filter_size1[2]*filter_size2[2])))+6];
            anchor_gridY = anchorY[int(((ci-num_filters[0]-num_filters[1])/(filter_size1[2]*filter_size2[2])))+6];
            stride = 32;
        }

        cx = (float)(cx*2-0.5+gridX)*stride;
        cy = (float)(cy*2-0.5+gridY)*stride;
        w = w*2*w*2*anchor_gridX;
        h = h*2*h*2*anchor_gridY;

        values[c]   = cx;
        values[c+1] = cy;
        values[c+2] = w;
        values[c+3] = h;
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    cout << "Time taken for DETECT : " << microseconds << endl;
    std::ofstream out;
    out.open( argv[2], std::ios::out | std::ios::binary);
    for(int i = 0; i < values.size(); i++)
        out.write( reinterpret_cast<const char*>( &values[i] ), sizeof( float ));
    out.close();

    return 0;
}
