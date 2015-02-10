#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

vector<Point> mousev, kalmanv;
Point mouseCoords(-1, -1);
Point lastMouse;

void onMouse(int event, int x, int y, int flags, void *param);
void drawX(Mat &img, Point &center, Scalar color, int d);
void drawPath(Mat &img, vector<Point> &points, Scalar color);
void initKalmanFilter(KalmanFilter &KF,
                      Point &xy,
                      Mat &transitionMatrix,
                      double processVariance,
                      double measurementVariance,
                      double postErrorVariance);
void printKF(KalmanFilter &KF);

int main(int argc, char *const argv[])
{
  Mat img(500, 500, CV_8UC3);
  KalmanFilter KF(4, 2, 0);
  Mat measurement = Mat::zeros(2, 1, CV_32F);
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
    initKalmanFilter(KF, mouseCoords, tm, procVar, measVar, errVar);

    mousev.clear();
    kalmanv.clear();

    while (true)
    {
      Mat prediction = KF.predict();
      Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

      measurement = (Mat_<float>(2, 1) << mouseCoords.x, mouseCoords.y);

      Point measPt(measurement.at<float>(0), measurement.at<float>(1));
      mousev.push_back(measPt);

      Mat estimated = KF.correct(measurement);
      Point statePt(estimated.at<float>(0), estimated.at<float>(1));
      kalmanv.push_back(statePt);

      // Visualization
      img = Scalar::all(0);
      drawX(img, statePt, Scalar(255, 255, 255), 5);
      drawX(img, measPt, Scalar(0, 0, 255), 5);
      drawPath(img, mousev, Scalar(255, 255, 0));
      drawPath(img, kalmanv, Scalar(0, 255, 0));
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
  lastMouse = mouseCoords;
  mouseCoords.x = x;
  mouseCoords.y = y;
}

void drawX(Mat &img, Point &center, Scalar color, int d)
{
  line(img, Point(center.x - d, center.y - d),
       Point(center.x + d, center.y + d), color, 2, LINE_AA, 0);
  line(img, Point(center.x + d, center.y - d),
       Point(center.x - d, center.y + d), color, 2, LINE_AA, 0);
}

void drawPath(Mat &img, vector<Point> &points, Scalar color)
{
  for (size_t i = 0; i < points.size() - 1; i++)
  {
    line(img, points[i], points[i + 1], color, 2);
  }
}

void initKalmanFilter(KalmanFilter &KF,
                      Point &xy,
                      Mat &transitionMatrix,
                      double procVariance,
                      double measVariance,
                      double errorVariance)
{
  // Initialize state vector (x, y, vx, vy)
  KF.statePre.at<float>(0) = xy.x;
  KF.statePre.at<float>(1) = xy.y;
  KF.statePre.at<float>(2) = 0;
  KF.statePre.at<float>(3) = 0;

  KF.transitionMatrix = transitionMatrix;

  // Initialize measurement and covariance matrices (all scaled I matrices).
  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(procVariance));
  setIdentity(KF.measurementNoiseCov, Scalar::all(measVariance));
  setIdentity(KF.errorCovPost, Scalar::all(errorVariance));

  if (false)
    printKF(KF);
}

void printKF(KalmanFilter &KF)
{
  cout << "measurementMatrix = " << endl << " "  << KF.measurementMatrix << endl << endl;
  cout << "processNoiseCov = " << endl << " "  << KF.processNoiseCov << endl << endl;
  cout << "measurementNoiseCov = " << endl << " "  << KF.measurementNoiseCov << endl << endl;
  cout << "errorCovPost = " << endl << " "  << KF.errorCovPost << endl << endl;
}