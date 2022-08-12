#!/usr/bin/env python

from pathlib import Path
import argparse

from omegaconf import OmegaConf

from camera_calibrator import CameraCalibrator


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input-dir", type=Path, required=True, help="Input image directory"
    )
    ap.add_argument(
        "-c", "--config", type=Path, required=True, help="YAML configuration file"
    )
    ap.add_argument(
        "-o", "--output-dir", type=Path, required=False, help="Output location"
    )
    ap.add_argument("--display", action="store_true", help="Show images")
    return ap.parse_args()


def main():
    args = get_args()
    assert args.input_dir.is_dir(), f"Not a directory: {args.input_dir}"
    assert args.config.is_file(), f"Invalid config file: {args.config}"
    # assert args.output_dir.is_dir(), f"Not a directory: {args.output_dir}"

    image_files = args.input_dir.glob("*")
    image_names = [str(f) for f in image_files if f.is_file()]
    print("\n".join(image_names))

    conf = OmegaConf.load(str(args.config))
    print("\nConfiguration\n=============\n", OmegaConf.to_yaml(conf))

    cc = CameraCalibrator(image_names, conf)

    cc.run()

    print(args.output_dir)
    if args.output_dir and args.output_dir.is_dir():
        cc.save_corrected_images(args.output_dir)


if __name__ == "__main__":
    main()
