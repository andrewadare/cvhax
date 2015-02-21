#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;

// Pixel gradient intensity threshold
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void *);

int main(int, char **argv)
{
  // Load source image and convert it to gray
  src = imread(argv[1], 1);

  // Convert image to gray and blur it
  cvtColor(src, src_gray, COLOR_BGR2GRAY);
  blur(src_gray, src_gray, Size(3, 3));

  // Create Window
  const char *source_window = "Source";
  namedWindow(source_window, WINDOW_AUTOSIZE);
  imshow(source_window, src);

  createTrackbar("Canny threshold:", "Source", &thresh, max_thresh, thresh_callback);
  thresh_callback(0, 0);

  waitKey(0);
  return (0);
}

void thresh_callback(int, void *)
{
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  // Detect edges using canny. Result is a single-channel 8 bit binary image.
  Canny(src_gray, canny_output, thresh, thresh * 2, 3);
  // threshold(src_gray, canny_output, thresh, 255, THRESH_BINARY );

  // Dilation and erosion structuring elements
  int ds = 15, es = 12; // Sizes of morphology elements (px)
  Mat dkern = getStructuringElement(cv::MORPH_ELLIPSE,
                                    Size(2*ds + 1, 2*ds+1),
                                    Point(ds, ds));
  Mat ekern = getStructuringElement(cv::MORPH_ELLIPSE,
                                    Size(2*es + 1, 2*es+1),
                                    Point(es, es));

  dilate(canny_output, canny_output, dkern);
  erode(canny_output, canny_output, ekern);

  imshow("Result of Canny edge detection", canny_output);

  // Find contours. Image is modified in place.
  int retMode = RETR_EXTERNAL; // RETR_TREE; // RETR_EXTERNAL;
  findContours(canny_output, contours, hierarchy, retMode, CHAIN_APPROX_SIMPLE);

  Mat rgbcont = cv::Mat(canny_output.size(), CV_8UC3);
  cv::cvtColor(canny_output, rgbcont, COLOR_GRAY2RGB);

  Scalar green(0,255,0);
  for (size_t i = 0; i < contours.size(); i++)
    drawContours(rgbcont, contours, (int)i, green, 2, 8, hierarchy, 0);

  // For some reason the edges are not being drawn.
  imshow("Edges and contours", rgbcont);

  // Find the convex hull object for each contour
  vector<vector<Point> > hulls;
  vector<Rect> boundRect;
  for (size_t i = 0; i < contours.size(); i++)
  {
    vector<Point> hull;
    convexHull(Mat(contours[i]), hull, false);
    if (contourArea(hull) > 100)
    {
      hulls.push_back(hull);
      boundRect.push_back(boundingRect(Mat(hull)));
    }
  }

  // Get the center of mass of each convex hull
  vector<Point2f> mc(hulls.size());
  for (size_t i = 0; i < hulls.size(); i++)
  {
    Moments mu = moments(hulls[i], false);
    mc[i] = Point2f(static_cast<float>(mu.m10 / mu.m00) ,
                    static_cast<float>(mu.m01 / mu.m00));
  }

  Mat drawing = src.clone(); //Mat::zeros(canny_output.size(), CV_8UC3);

  // Draw contours
  // for (size_t i = 0; i < contours.size(); i++)
  // {
  //   Scalar white = Scalar(255,255,255);
  //   drawContours(drawing, contours, (int)i, white, 2, 8, hierarchy, 0, Point());
  // }

  // Draw convex hulls, bounding rectangles, and CM points on image.
  for (size_t i = 0; i < hulls.size(); i++)
  {
    Scalar color = Scalar(rng.uniform(0, 255),
                          rng.uniform(0, 255),
                          rng.uniform(0, 255));
    rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
    drawContours(drawing, hulls, (int)i, color, 2, 8, hierarchy, 0, Point());
    circle(drawing, mc[i], 4, color, -1, 8, 0);
  }

  // Show in a window
  namedWindow("Contours", WINDOW_AUTOSIZE);
  imshow("Contours", drawing);
}
