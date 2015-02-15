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
  TrackedPoint(int x0in, int y0in, int vx0, int vy0);
  ~TrackedPoint() {}
  void stepTo(Point &p);
  void stay();
  void coast();
  Point nearestPoint(list<Point> &l, bool pop = true);
  int x0,y0,x,y,kx,ky,vx,vy,kvx,kvy;
  double v; // mean velocity = arc length of obsTail / obsTail.size();
  Point xyMin, xyMax;
  deque<Point> obsTail, kfTail;
  long lifetime;
  bool inBounds;
  int nTailPoints;
  int nCoasts, maxNCoasts;
  double coastLength, maxCoastLength;
  KalmanFilter kf;
};

bool inBounds(Point &p, Mat &img)
{
  return (p.x >= 0 && p.x < img.cols && p.y >= 0 && p.y < img.rows);
}

bool outOfBounds(const TrackedPoint &t)
{
  return !t.inBounds;
}

bool coastedTooLong(const TrackedPoint &t)
{
  return t.nCoasts > t.maxNCoasts;
}

bool coastedTooFar(const TrackedPoint &t)
{
  return t.coastLength > t.maxCoastLength;
}

TrackedPoint::TrackedPoint(int x0in, int y0in, int vx0, int vy0) :
  x0(x0in),                  // Position at initialization
  y0(y0in),
  x(x0in),                   // Measured position
  y(y0in),
  kx(x0in),                  // Kalman position
  ky(y0in),
  vx(vx0),                   // Step velocity dx/dt (= step size).
  vy(vy0),
  kvx(vx0),                  // Kalman velocity
  kvy(vy0),
  v(0.0),                    // Historical speed
  xyMin(Point(0,0)),         // Boundaries
  xyMax(Point(1000,1000)),
  lifetime(0),
  inBounds(true),
  nTailPoints(50),
  nCoasts(0), // Number of frames this point "lost the lock" and coasted.
  maxNCoasts(20),
  coastLength(0.),           // Coast distance so far
  maxCoastLength(1000.),     // Allowable coast distance
  kf(KalmanFilter(2,2))      // 2 state pars, 2 measurement inputs (both x,y)
{
  Point xy(x, y);
  double procVar = 1e-4, measVar = 1e-3, errVar = 0.1;

  kf.transitionMatrix = Mat::eye(2, 2, CV_32F);
  kf.statePre.at<float>(0) = xy.x;
  kf.statePre.at<float>(1) = xy.y;

  // Measurement and covariance matrices (all scaled identity matrices).
  setIdentity(kf.measurementMatrix);
  setIdentity(kf.processNoiseCov, Scalar::all(procVar));
  setIdentity(kf.measurementNoiseCov, Scalar::all(measVar));
  setIdentity(kf.errorCovPost, Scalar::all(errVar));
}

void TrackedPoint::stay()
{
  Point here(x,y);
  stepTo(here);
}

void TrackedPoint::coast()
{
  Point next(x + kvx, y + kvy);
  coastLength += norm(next - Point(x,y));
  stepTo(next);
  nCoasts++;
}

void TrackedPoint::stepTo(Point &p)
{
  lifetime++;

  x = p.x;
  y = p.y;

  obsTail.push_back(Point(x, y));
  if (obsTail.size() > nTailPoints)
    obsTail.pop_front();

  // Compute "historical" velocity of this point as the track length / # steps
  vector<Point> tail(obsTail.begin(), obsTail.end());
  v = arcLength(Mat(tail), false) / tail.size();

  if (x < xyMin.x || x > xyMax.x || y < xyMin.y || y > xyMax.y)
  {
    inBounds = false;
    return;
  }
  else
  {
    Mat measurement = (Mat_<float>(2, 1) << x, y);
    kf.predict();
    kf.correct(measurement);

    // Position and velocity of Kalman state
    kx = kf.statePost.at<float>(0);
    ky = kf.statePost.at<float>(1);

    kfTail.push_back(Point(kx, ky));
    size_t N = kfTail.size();
    if (N > nTailPoints)
      kfTail.pop_front();

    if (N > 2)
    {
      kvx = (kx - kfTail[N-3].x)/2;
      kvy = (ky - kfTail[N-3].y)/2;
    }
  }
}

Point TrackedPoint::nearestPoint(list<Point> &l, bool pop)
{
  Point here(x,y);
  double minDist = 1e15;
  Point nearest(-1, -1);
  for (list<Point>::iterator it = l.begin(); it != l.end(); ++it)
  {
    double dist = cv::norm(*it - here);
    if (dist < minDist)
    {
      minDist = dist;
      nearest = *it;
    }
  }
  if (pop)
    l.remove(nearest);

  return nearest;
}

void addSimPoint(list<TrackedPoint> &l, Mat &img)
{
  int x0 = 0, y0 = rng.uniform(50, img.rows-50);
  int vx0 = rng.uniform(3, 8), vy0 = rng.uniform(-1,1);
  TrackedPoint t(x0, y0, vx0, vy0);
  t.xyMin = Point(0, 0);
  t.xyMax = Point(img.cols-1, img.rows-1);
  l.push_back(t);
}

void addPoint(list<TrackedPoint> &l, Point &p, Mat &img)
{
  TrackedPoint t(p.x, p.y, 0, 0);
  t.x0 = p.x;
  t.y0 = p.y;
  t.xyMin = Point(0, 0);
  t.xyMax = Point(img.cols-1, img.rows-1);
  t.maxCoastLength = 0.1*norm(Point(img.cols, img.rows));
  l.push_back(t);
}

Point addNoise(int x, int y, double xsigma, double ysigma)
{
  return Point(x + rng.gaussian(xsigma), y + rng.gaussian(ysigma));
}

int main(int argc, char *const argv[])
{
  int npts = 10;

  // Simulated points following a smooth path (up to process noise).
  // These points represent the underlying physical truth.
  list<TrackedPoint> simPts;

  // Position measurements in each frame and their noise parameters.
  list<Point> xym;
  double xsigma = 4.0, ysigma = 4.0;

  // Observed points representing our best effort to reconstruct trajectories
  // from noisy position measurements.
  list<TrackedPoint> obsPts;

  Mat img(500, 1000, CV_8UC3);
  Rect border(0, 0, img.cols-1, img.rows-1);
  string windowName("Point tracker");
  namedWindow(windowName);
  char code = char(-1);
  list<TrackedPoint>::iterator it;

  while (true)
  {
    // Generate simulated points.
    while (simPts.size() < npts)
      addSimPoint(simPts, img);

    // Advance simulated points. Simulate position measurements.
    for (it = simPts.begin(); it != simPts.end(); ++it)
    {
      int x = it->x0 + it->vx * it->lifetime;
      int y = it->y0 + it->vy * it->lifetime;
      Point newxy(x,y);
      it->stepTo(newxy);

      // Collect less-than-perfect position measurements.
      if (it->inBounds)
      {
        Point meas = addNoise(it->x, it->y, xsigma, ysigma);
        if (inBounds(meas, img))
          xym.push_back(Point(meas.x, meas.y));
      }
    }
    simPts.remove_if(outOfBounds);

    // Step observations to the nearest available measurement.
    // If no nearby measurement is found in this frame, coast.
    // In real use, "nearby" might be ~ 2x typical convex hull radii.
    for (it = obsPts.begin(); xym.size() > 0 && it != obsPts.end(); ++it)
    {
      Point p = it->nearestPoint(xym);
      double dist = cv::norm(p - Point(it->x, it->y));

      if (dist < 50)
        it->stepTo(p);
      else
        it->coast();
    }
    obsPts.remove_if(outOfBounds);
    obsPts.remove_if(coastedTooLong);
    obsPts.remove_if(coastedTooFar);

    // Create new tracked observations from any remaining measurements
    for (list<Point>::iterator ip = xym.begin(); ip != xym.end(); ++ip)
    {
      addPoint(obsPts, *ip, img);
      xym.remove(*ip);
    }

    // Draw simulated points
    img = Scalar::all(0);
    for (it = simPts.begin(); it != simPts.end(); ++it)
    {
      deque<Point> ot = it->obsTail;
      for (size_t j = 0; j < ot.size() - 1; j++)
        line(img, ot[j], ot[j + 1], Scalar(100,100,100), 2);
      circle(img, it->obsTail.back(), 4, Scalar(100,100,100), -1, 8, 0);
    }

    // Debug: draw this frame's measurements.
    // If everything is working, xym is empty and no green points are visible.
    for (list<Point>::iterator ip = xym.begin(); ip != xym.end(); ++ip)
      circle(img, *ip, 4, Scalar(0,255,0), -1, 8, 0);

    // Draw observed points
    for (it = obsPts.begin(); it != obsPts.end(); ++it)
    {
      if (it->lifetime < 2) continue;

      deque<Point> ot = it->obsTail;
      for (size_t j = 0; j < ot.size() - 1; j++)
        line(img, ot[j], ot[j + 1], Scalar(0,0,255), 2);
      circle(img, it->obsTail.back(), 4, Scalar(0,0,255), -1, 8, 0);

      deque<Point> kt = it->kfTail;
      for (size_t j = 0; j < kt.size() - 1; j++)
        line(img, kt[j], kt[j + 1], Scalar(255,255,255), 2);
      circle(img, it->kfTail.back(), 4, Scalar(255,255,255), -1, 8, 0);
    }

    rectangle(img, border.tl(), border.br(), Scalar(255, 255, 255), 2, 8, 0);
    imshow(windowName, img);
    code = (char)waitKey(30);

    if (code == 27 || code == 'q')
      break;
  }

  return 0;
}
