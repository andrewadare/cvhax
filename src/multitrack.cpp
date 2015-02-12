#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <list>

using namespace cv;
using namespace std;

RNG rng(12345);

class TrackedPoint
{
public:
  TrackedPoint(int x0, int y0, int vx0, int vy0);
  ~TrackedPoint() {}
  void step();
  double x,y,vx,vy;
  Point xyMin, xyMax;
  deque<Point> obsTail;
  deque<Point> kfTail;
  long lifetime;
  bool inBounds;
  KalmanFilter kf;
};

bool outOfBounds(const TrackedPoint &t)
{
  return !t.inBounds;
}

TrackedPoint::TrackedPoint(int x0, int y0, int vx0, int vy0) :
  x(x0),
  y(y0),
  vx(vx0),
  vy(vy0),
  xyMin(Point(0,0)),
  xyMax(Point(1000,1000)),
  lifetime(0),
  inBounds(true),
  kf(KalmanFilter(4,2,0))
{
  Point xy(x, y);
  double procVar = 1e-4, measVar = 1e-3, errVar = 0.1;

  kf.transitionMatrix = Mat::eye(4, 4, CV_32F);
  kf.statePre.at<float>(0) = xy.x;
  kf.statePre.at<float>(1) = xy.y;
  kf.statePre.at<float>(2) = 0;
  kf.statePre.at<float>(3) = 0;

  // Measurement and covariance matrices (all scaled identity matrices).
  setIdentity(kf.measurementMatrix);
  setIdentity(kf.processNoiseCov, Scalar::all(procVar));
  setIdentity(kf.measurementNoiseCov, Scalar::all(measVar));
  setIdentity(kf.errorCovPost, Scalar::all(errVar));
}

void TrackedPoint::step()
{
  double xsigma = 4.0, ysigma = 4.0;
  int nTailPoints = 50;
  x += vx + rng.gaussian(xsigma);
  y += vy + rng.gaussian(ysigma);
  obsTail.push_back(Point(x, y));
  if (obsTail.size() > nTailPoints)
    obsTail.pop_front();

  if (x < xyMin.x || x > xyMax.x || y < xyMin.y || y > xyMax.y)
  {
    inBounds = false;
    return;
  }
  else
  {
    Mat prediction = kf.predict();
    Mat measurement = (Mat_<float>(2, 1) << x, y);
    Mat kfState = kf.correct(measurement);

    Point p(kfState.at<float>(0), kfState.at<float>(1));
    kfTail.push_back(Point(p.x, p.y));
    if (kfTail.size() > nTailPoints)
      kfTail.pop_front();
  }
}

void addPoint(list<TrackedPoint> &l, Mat &img)
{
  int x0 = 0, y0 = rng.uniform(50, img.rows-50);
  int vx0 = rng.uniform(3, 8), vy0 = rng.uniform(-1,1);
  TrackedPoint t(x0, y0, vx0, vy0);
  t.xyMin = Point(0, 0);
  t.xyMax = Point(img.cols-1, img.rows-1);
  l.push_back(t);
}

int main(int argc, char *const argv[])
{
  int npts = 10;

  list<TrackedPoint> tps;

  Mat img(500, 1000, CV_8UC3);
  string windowName("Point tracker");
  namedWindow(windowName);
  char code = char(-1);

  for (size_t i=0; i<npts; ++i)
  {
    addPoint(tps, img);
  }

  while (true)
  {
    int shortage = npts - tps.size();
    while (shortage > 0)
    {
      addPoint(tps, img);
      shortage--;
    }

    for (list<TrackedPoint>::iterator it = tps.begin(); it != tps.end(); ++it)
    {
      it->step();
      TrackedPoint p = *it;
    }
    tps.remove_if(outOfBounds);


    // Visualization
    img = Scalar::all(0);

    for (list<TrackedPoint>::iterator it = tps.begin(); it != tps.end(); ++it)
    {
      deque<Point> ot = it->obsTail;
      for (size_t j = 0; j < ot.size() - 1; j++)
        line(img, ot[j], ot[j + 1], Scalar(0,0,255), 2);
      circle(img, it->obsTail.back(), 4, Scalar(0,0,255), -1, 8, 0);

      deque<Point> kt = it->kfTail;
      for (size_t j = 0; j < kt.size() - 1; j++)
        line(img, kt[j], kt[j + 1], Scalar(255,255,255), 2);
      circle(img, it->kfTail.back(), 4, Scalar(255,255,255), -1, 8, 0);
    }

    imshow(windowName, img);
    code = (char)waitKey(30);

    if (code == 27 || code == 'q')
      break;
  }

  return 0;
}
