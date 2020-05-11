#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

Mat total_mask ;
vector<string> classes;

void postprocess(Mat& image, const vector<Mat>& outs);

int main(int argc, char** argv)
{
    // Load names of classes
    string classesFile = "data/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String textGraph = "data/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    String modelWeights = "data/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

    // Load the network
    Net net = readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat image = imread("img/image4.jpeg") ; 
    total_mask = Mat(Size(image.rows, image.cols), CV_8UC1) ;
    total_mask.setTo(Scalar(0)) ;
    if (image.channels() == 4) cvtColor(image, image, COLOR_BGRA2BGR);
    Mat blur, orig = image.clone() ;
    imshow("orig", image) ;

    Mat blob ;
    blobFromImage(image, blob, 1.0, Size(image.cols, image.rows), Scalar(), true, false);
    
    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output from the output layers
    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";
    vector<Mat> outs;
    net.forward(outs, outNames);
    
    // Extract the bounding box and mask for each of the detected objects
    postprocess(image, outs);

    GaussianBlur(image, blur, Size(9,9), 5, 5, BORDER_DEFAULT) ;
    orig.copyTo(blur, total_mask) ;
    
    imshow("total mask", total_mask) ;
    imshow("result", blur) ;

    imwrite("img/out4.png", blur) ;

    waitKey(0) ;
    return 0;
}

// For each image, extract the bounding box and mask for each detected object
void postprocess(Mat& image, const vector<Mat>& outs)
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];
    
    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];
    
    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            // Extract the bounding box
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(image.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(image.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(image.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(image.rows * outDetections.at<float>(i, 6));
            
            left = max(0, min(left, image.cols - 1));
            top = max(0, min(top, image.rows - 1));
            right = max(0, min(right, image.cols - 1));
            bottom = max(0, min(bottom, image.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);
            
            // Extract the mask for the object
            Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));
            
            resize(objectMask, objectMask, Size(box.width, box.height));
            Mat mask = (objectMask > maskThreshold);

            vector<Mat> contours;
            Mat hierarchy;
            mask.convertTo(mask, CV_8U);
            findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

            mask.copyTo(total_mask(box), mask) ;
        }
    }
}