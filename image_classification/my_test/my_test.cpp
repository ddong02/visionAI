#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;
int main(void)
{
// Load an image
Mat img;
//img = imread("cat1.jpg", IMREAD_COLOR);
img = imread("../data/test1/dog.jpg", IMREAD_COLOR);

if (img.empty()) {
cerr << "Image load failed!" << endl;
return -1;
}
// Load network
Net net = readNet("../data/convert_pth.onnx"); 

if (net.empty()) {
cerr << "Network load failed!" << endl;
return -1;
}

// Load class names
vector<String> classNames={"cat","dog"};

// Inference
Mat inputBlob = blobFromImage(img, 1 / 255.f, Size(256, 256));
net.setInput(inputBlob);
Mat prob = net.forward();

// Check results & Display
double maxVal;
Point maxLoc;
minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);

String str = format("%s (%4.2lf%%)", classNames[maxLoc.x].c_str(), maxVal * 100);
putText(img, str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255));
imshow("img", img);

waitKey();
return 0;
}