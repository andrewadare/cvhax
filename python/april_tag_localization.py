#!/usr/bin/env python

"""Single-image tag localization example

Summary:
 - 3D scene with April tags in a grid on the floor
 - A synthetic image is captured using a simulated camera with known pose
and intrinsics.
 - OpenCV Aruco detector finds tags in the camera's screenshot.
 - Camera pose is estimated from detections and compared to ground truth.

Dependencies:
pip install pyvista opencv-python matplotlib

Issues:
 - Pose estimates disagree with ground truth more than expected in the range
 coordinate. I suspect an inconsistency between the projection model used in 
 image capture vs. pose estimation.
 Troubleshooting: 
 - examine f in setup_plotter [px]
 - examine focal length set by pose assignment (position - focal point) [m]
 - what is their theoretical relationship? In a real camera, this comes from
 calibration. Below, I set f = diagonal image size / 2.

    f = 0.5 * imsize / tan(fov/2)

"""
import math
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

# For drawing coord frame axes
rgb_colormap = ListedColormap(["red", "lawngreen", "dodgerblue"])

# Tag families or "dictionaries" available in OpenCV's aruco module
ARUCO_CODES = {k: v for k, v in cv2.aruco.__dict__.items() if k.startswith("DICT_")}


@dataclass
class Tag:
    """Square fiducial tag.

    Attributes
    ----------

    id: detectable integer identifer

    size: edge length [m]

    image: for texture mapping onto a mesh

    codebook: tag family / dictionary

    pose: 4x4 homogeneous transform in world frame

    pose_camera: PnP estimate of tag pose in camera frame
    """

    id: int
    size: float
    image: np.ndarray
    codebook: str
    pose: np.ndarray
    pose_camera: np.ndarray = field(default_factory=lambda: np.eye(4))

    def corners(self) -> np.ndarray:
        """Return tag corners in world frame.
        Point ordering follows OpenCV: TL, TR, BR, BL
        """
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        return (R @ tag_corners_local(self.size).T).T + t


def create_tag(codebook: str, tag_id: int, tag_width: int, border: int) -> np.ndarray:
    """Create an Aruco or April tag image using the cv2.aruco module.

    Usage
    =====
    Tag codebook options are discoverable in cv2.aruco.__dict__ as "DICT_*".

    To make a 400x400 px image of a 300x300 April 16h5 tag with id 0 and a
    50 px white border, do

    tag = create_tag("DICT_APRILTAG_16h5", 0, 300, 50)
    """

    if codebook not in ARUCO_CODES:
        raise LookupError(
            f"Tag dictionary '{codebook}' not found. Available options:\n"
            f"{' '.join(ARUCO_CODES.keys())}"
        )

    code = cv2.aruco.getPredefinedDictionary(ARUCO_CODES[codebook])
    tag = cv2.aruco.generateImageMarker(code, tag_id, tag_width)

    w = tag_width + 2 * border
    im = np.full((w, w), 255, dtype=np.uint8)
    im[border:-border, border:-border] = tag

    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def add_tag_mesh(tag: Tag, plotter: pv.Plotter):
    texture = pv.numpy_to_texture(tag.image)
    plane = pv.Plane(i_size=tag.size, j_size=tag.size)
    plane.transform(tag.pose)
    plotter.add_mesh(plane, texture=texture)

    # TODO figure out texture mapping to front face only.
    # For now, cover the back with another white plane shifted back 1mm
    offset = np.eye(4)
    offset[2, 3] = -0.001
    back = pv.Plane(i_size=tag.size, j_size=tag.size)
    back.transform(tag.pose @ offset)
    plotter.add_mesh(back, color="white")


def setup_plotter(image_size: np.ndarray, fov_degrees: float):
    """Create a PyVista plotter and a standard pinhole camera matrix based on
    the provided parameters.

    tan(fov/2) = image_size/(2f)
    cx, cy = image_size/2
    skew = radial distortion = 0.
    """
    plotter = pv.Plotter(off_screen=True, window_size=image_size)

    # Window center in view coords between [-1, 1]. If principal point is taken to be image_size/2,
    # window center is 0, 0. More generally,
    # wx = -2 * (cx - w / 2) / w
    # wy = 2 * (cy - h / 2) / h
    plotter.camera.SetWindowCenter(0, 0)

    # Camera field of view
    plotter.camera.SetViewAngle(fov_degrees)

    plotter.show_axes()  # origin
    plotter.set_background("slategray")

    # Matrix of intrinsic camera parameters
    cx, cy = image_size / 2
    fx, fy = 0.5 * image_size / math.tan(math.radians(fov_degrees / 2))
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return plotter, camera_matrix


def tag_corners_local(tag_size: float) -> np.ndarray:
    """Corners of tag with respect to its center.
    Follows OpenCV ordering TL, TR, BR, BL.
    """
    a: float = tag_size / 2
    return np.array([(-a, a, 0), (a, a, 0), (a, -a, 0), (-a, -a, 0)])


def generate_floor_tags(
    room_dimensions: np.array,
    tag_spacing: float,
    tag_size: float,
    tag_codebook: str,
) -> dict[int, Tag]:
    """Create tags in a grid pattern on the floor of a room with room_dimensions."""
    tags: dict[int, Tag] = dict()  # id => Tag
    id = 0
    for x in np.arange(0, room_dimensions[0], tag_spacing):
        for y in np.arange(0, room_dimensions[1], tag_spacing):
            im = create_tag(tag_codebook, id, 300, 50)
            T = np.eye(4)
            T[:3, 3] = (x, y, 0)
            tags[id] = Tag(id, tag_size, im, tag_codebook, T)
            id += 1
    return tags


def simulate_camera_localization(image_file: str):
    room_dimensions = np.array([15.0, 12.0, 4.0])  # meters
    tag_spacing = 5  # meters
    tag_size = 1  # meters
    tag_codebook = "DICT_APRILTAG_16h5"
    plotter, camera_matrix = setup_plotter(
        image_size=np.array([1440, 1080]), fov_degrees=100
    )

    tags = generate_floor_tags(room_dimensions, tag_spacing, tag_size, tag_codebook)
    for tag_id, tag in tags.items():
        add_tag_mesh(tag, plotter)

    # Set the camera pose, which is our ground truth reference for localization.
    # Updating this has no effect on the pyvista camera.model_transform_matrix M.
    # Instead, use the VTK method Camera.GetModelViewTransformMatrix to get the
    # model-to-camera transform, which is equivalent to the world-to-camera
    # transform cTw if M = I(4).
    # So if I don't touch M, the camera pose wTc I'm after is inv(cTw).
    plotter.camera.position = np.random.random(3) * room_dimensions / 2
    plotter.camera.focal_point = (*room_dimensions[:2] / 2, -5 * np.random.uniform())

    # Simpler deterministic example...
    # plotter.camera.position = (*room_dimensions[:2] / 2, 5)
    # plotter.camera.up = (0, 1, 0)
    # plotter.camera.focal_point = (*room_dimensions[:2] / 2, -100)

    # cTw is the world-to-camera transform: pc = cTw * pw.
    cTw_vtk = plotter.camera.GetModelViewTransformMatrix()

    # No built-in conversion from VTK matrix to Numpy array?
    cTw = np.eye(4)
    for i in range(4):
        for j in range(4):
            cTw[i, j] = cTw_vtk.GetElement(i, j)

    # Camera pose
    wTc = np.linalg.inv(cTw)
    np.savetxt(Path(image_file).stem + ".txt", wTc)

    plotter.add_light(pv.Light(position=room_dimensions / 2, light_type="scene light"))
    # plotter.show() # no effect if off_screen=True

    # Save and also return the simulated image, which must be contiguous.
    # https://stackoverflow.com/a/50128836
    image = plotter.screenshot(image_file, return_img=True)
    image = np.ascontiguousarray(image, dtype=np.uint8)

    # Point detection and pose estimation from here onward
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.minMarkerDistanceRate = 0.02  # default 0.05
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_CODES[tag_codebook]),
        aruco_params,
    )

    # Each marker has shape (1,4,2)
    markers, ids, rejected = detector.detectMarkers(image)

    if ids is None:
        print("No tags detected.")
        return

    ids = ids.ravel()
    print(f"Detected {len(markers)} markers: {ids}")

    # No lens distortion modeled here, but include for generality.
    distortion = np.zeros(5)

    # Pose estimation per marker in camera frame for visualization.
    # TODO: Pose estimates occasionally get screwed up for poor detections.
    # TODO: Average over tag-wise camera pose estimates to identify outliers.
    for id, marker in zip(ids, markers):
        uv = cv2.undistortPoints(marker, camera_matrix, distortion).squeeze()
        retval, r, t = cv2.solvePnP(
            tag_corners_local(tags[id].size),
            uv,
            np.eye(3),
            np.zeros(5),
            cv2.SOLVEPNP_ITERATIVE,
        )
        R, _ = cv2.Rodrigues(r)
        tags[id].pose_camera[:3, :3] = R
        tags[id].pose_camera[:3, 3] = t.squeeze()

        cv2.drawFrameAxes(image, camera_matrix, distortion, r, t, 0.9 * tags[0].size)

    cv2.aruco.drawDetectedMarkers(image, markers, ids)
    cv2.imwrite(image_file, image)  # TODO: this clobbers the first screenshot

    # Joint estimation over all observed tags
    distorted_image_points = np.vstack([m.squeeze() for m in markers])
    image_points = cv2.undistortPoints(
        distorted_image_points, camera_matrix, distortion
    ).squeeze()
    world_points = np.vstack([tags[id].corners() for id in ids])

    retval, r, t = cv2.solvePnP(
        world_points,
        image_points,
        np.eye(3),
        np.zeros(5),
        cv2.SOLVEPNP_ITERATIVE,
    )
    R, _ = cv2.Rodrigues(r)
    t = t.ravel()

    # Modify PnP estimate to handle 2 coordinate interpretation differences between
    # geometric computer vision conventions and our desired representation:
    # 1. Passive vs active rotation: Use R.T instead of R
    # 2. A basis change B in SO(3) is required to reinterpret the xyz coordinates.
    #    OpenGL xyz = right-up-back ==> Camera right-down-fwd.
    #    If the world coordinate frame followed some other convention such as ROS REP 103,
    #    B would be different than below.
    # Issue 2 could also be handled by representing the tag poses in camera coordinates
    # from the beginning, then B below would be I(3).
    B = np.diag([1, -1, -1])
    pnp_pose = np.eye(4)
    pnp_pose[:3, :3] = R.T @ B
    pnp_pose[:3, 3] = -R.T @ t

    print("PnP pose")
    print(pnp_pose)

    print("Ground truth")
    print(wTc)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    simulate_camera_localization("andrew.png")
