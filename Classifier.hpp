//
//  Classifier.hpp
//  casperEye
//
//  Created by Nguyen Duy Vu on 8/1/16.
//  Copyright © 2016 Nguyen Duy Vu. All rights reserved.
//

#ifndef Classifier_hpp
#define Classifier_hpp
#define CPU_ONLY

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>
#include <memory>

using namespace caffe;  // NOLINT(build/namespaces)

struct predict{int label; float percent;};

class Classifier {

public:
    Classifier();
    void load(const string& model_file,
               const string& trained_file,
               const string& mean_file);
    
    predict Classify(const cv::Mat& img, int N);
    
private:
    void SetMean(const string& mean_file);
    
    std::vector<float> Predict(const cv::Mat& img);
    
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    
    void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
    
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    
    std::vector<string> labels_;
    
    std::shared_ptr<caffe::Net<float> > net_;
};
#endif /* Classifier_hpp */

