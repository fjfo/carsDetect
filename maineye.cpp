#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include "Classifier.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>          // For cout and cerr
#include <cstdlib>           // For atoi()
#include "config.h"

#define BUF_LEN 65540 // Larger than maximum UDP packet size

using namespace cv;
using namespace std;
using namespace cv::ml;

int main(int argc, char * argv[]) {   
    ::google::InitGoogleLogging(argv[0]);    	
    Classifier classifier;
    classifier.load(CAFFE_PROTO, CAFFE_MODEL,CAFFE_MEAN);            
    Mat img;    
    std::string dPath = "data/test1.jpeg";    
    Mat src = imread(dPath);
    resize(src, img, Size(256, 256), INTER_LINEAR);
    predict isCar = classifier.Classify(src,LABELS);                            
    cout << "Class:" << isCar.label << ", confidence:" << isCar.percent << endl;
     
    return 0;
}
