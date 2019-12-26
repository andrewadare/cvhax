import sys

import cv2
# import numpy as np


def main():
    cv2.namedWindow('video')
    vid = cv2.VideoCapture(0)

    # if not vid.isOpened():
    #     print("Error opening video stream or file")

    while vid.isOpened():
        ret, im = vid.read()
        if ret:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cv2.imshow('video', im)
            wk = cv2.waitKey(100)
            if wk == ord('q'):
                vid.release()
                cv2.destroyAllWindows()
                sys.exit(0)


if __name__ == '__main__':
    main()
