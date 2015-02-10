#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

Point mouseCoords(-1, -1);
vector<Point> mousePts, kfPts; // For drawing

void onMouse(int event, int x, int y, int flags, void *param);
void drawPath(Mat &img, vector<Point> &points, Scalar color);
void initKalmanFilter(KalmanFilter &kf,
                      Point &xy,
                      Mat &transitionMatrix,
                      double processVariance,
                      double measurementVariance,
                      double postErrorVariance);
void printKF(KalmanFilter &kf);

int main(int argc, char *const argv[])
{
  Mat img(500, 500, CV_8UC3);
  KalmanFilter kf(4, 2, 0);
  char code = (char) - 1;
  string windowName("Kalman Mouse Tracker");
  namedWindow(windowName);
  setMouseCallback(windowName, onMouse, 0);

  while (true)
  {
    if (mouseCoords.x < 0 || mouseCoords.y < 0)
    {
      imshow(windowName, img);
      waitKey(30);
      continue;
    }
    double procVar = 1e-4, measVar = 1e-3, errVar = 0.1;
    Mat tm = Mat::eye(4, 4, CV_32F);
    initKalmanFilter(kf, mouseCoords, tm, procVar, measVar, errVar);

    mousePts.clear();
    kfPts.clear();

    while (true)
    {
      // Predict, measure, correct.
      // Result is state vector estimated by Kalman Filter (x,y,vx,vy).
      Mat prediction = kf.predict();
      Mat measurement = (Mat_<float>(2, 1) << mouseCoords.x, mouseCoords.y);
      Mat kfstate = kf.correct(measurement);

      // Visualization
      mousePts.push_back(Point(mouseCoords.x, mouseCoords.y));
      kfPts.push_back(Point(kfstate.at<float>(0), kfstate.at<float>(1)));
      img = Scalar::all(0);
      drawPath(img, mousePts, Scalar(255, 255, 0));
      drawPath(img, kfPts, Scalar(0, 255, 0));
      circle(img, mousePts.back(), 4, Scalar(0,0,255), -1, 8, 0);
      circle(img, kfPts.back(), 4, Scalar(255,255,255), -1, 8, 0);
      imshow(windowName, img);
      code = (char)waitKey(100);

      if (code > 0)
        break;
    }
    if (code == 27 || code == 'q' || code == 'Q')
      break;
  }

  return 0;
}

void onMouse(int event, int x, int y, int flags, void *param)
{
  mouseCoords.x = x;
  mouseCoords.y = y;
}

void drawPath(Mat &img, vector<Point> &points, Scalar color)
{
  for (size_t i = 0; i < points.size() - 1; i++)
  {
    line(img, points[i], points[i + 1], color, 2);
  }
}

void initKalmanFilter(KalmanFilter &kf,
                      Point &xy,
                      Mat &transitionMatrix,
                      double procVariance,
                      double measVariance,
                      double errorVariance)
{
  // Initialize state vector (x, y, vx, vy)
  kf.statePre.at<float>(0) = xy.x;
  kf.statePre.at<float>(1) = xy.y;
  kf.statePre.at<float>(2) = 0;
  kf.statePre.at<float>(3) = 0;

  kf.transitionMatrix = transitionMatrix;

  // Initialize measurement and covariance matrices (all scaled I matrices).
  setIdentity(kf.measurementMatrix);
  setIdentity(kf.processNoiseCov, Scalar::all(procVariance));
  setIdentity(kf.measurementNoiseCov, Scalar::all(measVariance));
  setIdentity(kf.errorCovPost, Scalar::all(errorVariance));

  if (false)
    printKF(kf);
}

void printKF(KalmanFilter &kf)
{
  cout << "measurementMatrix = " << endl << " "  << kf.measurementMatrix << endl << endl;
  cout << "processNoiseCov = " << endl << " "  << kf.processNoiseCov << endl << endl;
  cout << "measurementNoiseCov = " << endl << " "  << kf.measurementNoiseCov << endl << endl;
  cout << "errorCovPost = " << endl << " "  << kf.errorCovPost << endl << endl;
}