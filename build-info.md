# Building OpenCV on Linux
Some notes:
1. I was fortunate to find a thorough (and recent!) [PyImageSearch Post](http://www.pyimagesearch.com/2015/10/26/how-to-install-opencv-3-on-raspbian-jessie/) for guidance.
2. I have OpenNI2 built already, and included here as a dependency. Edit the cmake flag below to modify.
3. I built OpenCV on my laptop (Dell XPS 9550, Ubuntu 16.04) and on a Raspberry Pi 3 (Raspbian Jesse 2016-03-18, kernel 4.1) using literally identical steps--exactly those listed here.
4. Always good to do `sudo apt update` first (but I forgot).

## Clone opencv and opencv_contrib from GitHub.
```
git clone https://github.com/Itseez/opencv.git
git clone https://github.com/Itseez/opencv_contrib.git
```
Put both of them on the desired version:
```
git tag -l # See what's available
git checkout -b 3.1.0 3.1.0
```
Be sure to include the tag twice--if not, HEAD will not be set to the correct point.

## Download dependencies
```
# Image i/o libs
sudo apt install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev

# Video i/o libs
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install libxvidcore-dev libx264-dev

# GTK
sudo apt install libgtk2.0-dev

# Fortran libs
sudo apt install libatlas-base-dev gfortran

# Python dev files, pip, virtualenv
sudo apt install python2.7-dev python3-dev
sudo apt install python-pip
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip

# Add this to .bashrc:
# virtualenv and virtualenvwrapper
if [ -f /usr/local/bin/virtualenvwrapper.sh ]; then
  export WORKON_HOME=$HOME/.virtualenvs
  source /usr/local/bin/virtualenvwrapper.sh
fi
```

## CMake configuration and compilation
```
$ mkdir ~/sw/OpenCV3.1
$ cd ~/sw/opencv
$ mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=~/sw/OpenCV3.1 \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D OPENCV_EXTRA_MODULES_PATH=~/sw/opencv_contrib/modules \
  -DWITH_OPENNI2=ON \
  -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF \
  -DWITH_IPP=OFF \
  -D BUILD_EXAMPLES=ON ..
```
If the output looks good, go for the build. Do `time make -j4` then `make install`. Build time on the Pi 3 was about 45 minutes.

## Making OpenCV available to user build systems
I put this in my .bashrc:
```
# Tell pkg-config where to find OpenCV headers and libs. It is the
# location of opencv.pc. To use: `pkg-config --cflags opencv` or `--libs`.
# Useful for simple command-line compilation of standalone programs
opencv_pc=/home/andrew/sw/OpenCV3.1/lib/pkgconfig
if [[ $PKG_CONFIG_PATH != *${opencv_pc}* ]]; then
    export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:$opencv_pc
fi
unset opencv_pc

# Provide location of OpenCVConfig.cmake for compiling with OpenCV using cmake.
export OpenCV_DIR=${HOME}/sw/OpenCV3.1/share/OpenCV
```

## Playing around with a few OpenCV examples on the Pi 3
Compute Delaunay triangulation of some random 2D points:
```
./cpp-example-delaunay2
```
Get an example image (this one is 470 x 600)
```
cp /usr/share/raspberrypi-artwork/raspberry-pi-logo.png ~/Pictures/
```
By the way, `gpicview` is a usable image viewer that ships with Raspbian.

In `~/sw/opencv/build/bin`, there are hundreds of examples, tutorials, and dataset files.

Here are some feature finders using the venerable Hough transform:
```
[bin]$ ./cpp-tutorial-HoughCircle_Demo ~/Pictures/raspberry-pi-logo.png
[bin]$ ./cpp-tutorial-HoughLines_Demo ~/Pictures/raspberry-pi-logo.png
```
This one steps through an image segmentation procedure:
```
[bin]$ ./cpp-tutorial-imageSegmentation ~/Pictures/raspberry-pi-logo.png
```
Compute contours from edges, then centroids, lengths, and areas:
```
[bin]$ ./cpp-tutorial-moments_demo ~/Pictures/raspberry-pi-logo.png
```
And on and on....
