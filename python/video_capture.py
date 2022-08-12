#!/usr/bin/env python

"""Displays video from a live camera or from a file. Typing 's' in the
video window saves a snapshot.
"""

import argparse
from pathlib import Path

import cv2

from common import display


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-s",
        "--video-source",
        default=0,
        help="Camera index or video file name. See docs for cv2.VideoCapture",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (for video) or directory (for images)",
    )
    return ap.parse_args()


def main():
    args = get_args()

    try:
        source = int(args.video_source)
    except ValueError:
        try:
            source = str(Path(args.video_source))
        except Exception as e:
            print("Warning: couldn't interpret video source")

    vid = cv2.VideoCapture(source)

    cv2.namedWindow("video")
    while vid.isOpened():
        ret, im = vid.read()
        if ret:
            display(im, "video", save_dir=args.output)


if __name__ == "__main__":
    main()
