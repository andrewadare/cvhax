#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

struct mouse_info_struct
{
  int x,y;
};
struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;

vector<Point> mousev, kalmanv;

void on_mouse(int event, int x, int y, int flags, void *param)
{
  last_mouse = mouse_info;
  mouse_info.x = x;
  mouse_info.y = y;
}

void drawX(Mat &img, Point &center, Scalar color, int d);
void drawPath(Mat &img, vector<Point> &points, Scalar color);

int main(int argc, char *const argv[])
{
  Mat img(500, 500, CV_8UC3);
  KalmanFilter KF(4, 2, 0);
  Mat state(4, 1, CV_32F);  // (x, y, v_x, v_y)
  Mat processNoise(4, 1, CV_32F);
  Mat measurement = Mat::zeros(2, 1, CV_32F);

  char code = (char)-1;
  string windowName("Kalman Mouse Tracker");
  namedWindow(windowName);

  setMouseCallback(windowName, on_mouse, 0);

  while (true)
  {
    if (mouse_info.x < 0 || mouse_info.y < 0)
    {
      imshow(windowName, img);
      waitKey(30);
      continue;
    }
    KF.statePre.at<float>(0) = mouse_info.x;
    KF.statePre.at<float>(1) = mouse_info.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    KF.transitionMatrix = Mat::eye(4,4,CV_32F);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    cout << "measurementMatrix = "<< endl << " "  << KF.measurementMatrix << endl << endl;
    cout << "processNoiseCov = "<< endl << " "  << KF.processNoiseCov << endl << endl;
    cout << "measurementNoiseCov = "<< endl << " "  << KF.measurementNoiseCov << endl << endl;
    cout << "errorCovPost = "<< endl << " "  << KF.errorCovPost << endl << endl;

    mousev.clear();
    kalmanv.clear();

    while (true)
    {
      Mat prediction = KF.predict();
      Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

      measurement = (Mat_<float>(2, 1) << mouse_info.x, mouse_info.y);

      Point measPt(measurement.at<float>(0),measurement.at<float>(1));
      mousev.push_back(measPt);

      Mat estimated = KF.correct(measurement);
      Point statePt(estimated.at<float>(0),estimated.at<float>(1));
      kalmanv.push_back(statePt);

      // Visualization
      img = Scalar::all(0);
      drawX(img, statePt, Scalar(255,255,255), 5);
      drawX(img, measPt, Scalar(0,0,255), 5);
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

void drawX(Mat &img, Point &center, Scalar color, int d)
{
  line(img, Point(center.x - d, center.y - d),
       Point(center.x + d, center.y + d), color, 2, LINE_AA, 0);
  line(img, Point(center.x + d, center.y - d),
       Point(center.x - d, center.y + d), color, 2, LINE_AA, 0);
}

void drawPath(Mat &img, vector<Point> &points, Scalar color)
{
  for (size_t i = 0; i < points.size()-1; i++)
  {
    line(img, points[i], points[i+1], color, 2);
  }
}
