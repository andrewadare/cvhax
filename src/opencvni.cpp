// Example: using OpenNI2 to capture a depth stream to OpenCV
// Based on example at https://gist.github.com/vins31/8a4af0a99392f45d5af4

#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;

static void colorizeDisparity(const Mat& gray, Mat& rgb, double maxDisp = -1.f, float S = 1.f, float V = 1.f)
{
  CV_Assert(!gray.empty());
  CV_Assert(gray.type() == CV_16UC1);

  if (maxDisp <= 0)
  {
    maxDisp = 0;
    minMaxLoc(gray, 0, &maxDisp);
  }

  rgb.create(gray.size(), CV_8UC3);
  rgb = Scalar::all(0);
  if (maxDisp < 1)
    return;

  for (int y = 0; y < gray.rows; y++)
  {
    for (int x = 0; x < gray.cols; x++)
    {
      unsigned short d = gray.at<unsigned short>(y, x);
      unsigned int H = ((unsigned short)maxDisp - d) * 240 / (unsigned short)maxDisp;

      unsigned int hi = (H / 60) % 6;
      float f = H / 60.f - H / 60;
      float p = V * (1 - S);
      float q = V * (1 - f * S);
      float t = V * (1 - (1 - f) * S);

      Point3f res;

      if (hi == 0)  //R = V,  G = t,  B = p
        res = Point3f(p, t, V);
      if (hi == 1)  // R = q, G = V,  B = p
        res = Point3f(p, V, q);
      if (hi == 2)  // R = p, G = V,  B = t
        res = Point3f(t, V, p);
      if (hi == 3)  // R = p, G = q,  B = V
        res = Point3f(V, q, p);
      if (hi == 4)  // R = t, G = p,  B = V
        res = Point3f(V, p, t);
      if (hi == 5)  // R = V, G = p,  B = q
        res = Point3f(q, p, V);

      uchar b = (uchar)(std::max(0.f, std::min(res.x, 1.f)) * 255.f);
      uchar g = (uchar)(std::max(0.f, std::min(res.y, 1.f)) * 255.f);
      uchar r = (uchar)(std::max(0.f, std::min(res.z, 1.f)) * 255.f);

      rgb.at<Point3_<uchar> >(y, x) = Point3_<uchar>(b, g, r);
    }
  }
}

int main()
{
  VideoCapture capture(CAP_OPENNI2_ASUS);
  if (!capture.isOpened())
  {
    std::cout << "***** Can not open a capture object." << std::endl;
    return -1;
  }
  for (;;)
  {
    Mat disparityMap;
    capture >> disparityMap;
    Mat colorDisparityMap;
    colorizeDisparity(disparityMap, colorDisparityMap, -1);
    Mat validColorDisparityMap;
    colorDisparityMap.copyTo(validColorDisparityMap, disparityMap != 0);
    imshow("depth map", validColorDisparityMap);
    if (waitKey(30) >= 0)
      break;
  }
  return 0;
}

// static float getMaxDisparity(VideoCapture& capture)
// {
//   const int minDistance = 400; // mm
//   float b = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_BASELINE);   // mm
//   float F = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH);   // pixels
//   return b * F / minDistance;
// }