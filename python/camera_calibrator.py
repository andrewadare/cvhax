from pathlib import Path

import cv2
import numpy as np
import omegaconf

from common import display


def hc(x: np.ndarray) -> np.ndarray:
    """Convert an array of points (as row vectors) from inhomogeneous to
    homogeneous coordinates. The returned points are in column vectors for
    pre-multiplication with a matrix.

    Parameters
    ----------
    x : ndarray - shape (m, n)
        Array of m row vectors in inhomogeneous coords, each of dimensionality n

    Returns
    -------
    ndarray - shape (n+1, m)
    """
    x = np.atleast_2d(x)

    return np.vstack([x.T, np.ones([1, x.shape[0]])])


def ic(x: np.ndarray, nan_to_num=True) -> np.ndarray:
    """Convert an array of points in column vectors from homogeneous to
    inhomogeneous coordinates. The returned array contains points as row
    vectors.

    Parameters
    ----------
    x : ndarray - shape (n+1, m)
        Array of m vectors in homogeneous coords, each of dimensionality n+1

    Returns
    -------
    ndarray - shape (m, n)
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(f"Expected numpy array, received {type(x)}")

    if x.size == 0:
        return x

    if x.ndim == 1:
        x = x[:, np.newaxis]

    with np.errstate(invalid="ignore"):
        x /= x[-1, :]

    x = x[:-1, :].T
    if nan_to_num:
        x = np.nan_to_num(x)  # np.inf -> large float; np.nan -> 0.0

    return x


class CameraCalibrator:
    def __init__(
        self,
        image_names: list[str],
        conf: omegaconf.dictconfig.DictConfig,
    ):
        self.image_names = image_names
        self.conf = conf

        # Specify corner positions in world coordinates.
        # The physical square edge length (square_size) affects
        # extrinsic calibration only.
        nx, ny = conf["checkerboard"]["num_inner_corners"]
        square_size = conf["checkerboard"]["square_size"]
        self.world_corners = np.zeros((nx * ny, 3), np.float32)
        grid = np.mgrid[0:nx, 0:ny]
        self.world_corners[:, :2] = square_size * grid.T.reshape(-1, 2)

        # Termination criteria used by cornerSubPix
        self.term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.world_point_list = []
        self.image_point_list = []
        self.subpix_window_size = (5, 5)
        self.K = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

    def get_corner_points(self, im):
        """Find the chessboard corners in pixel coordinates.

        Parameters
        ----------
        im : ndarray
            Grayscale OpenCV image
        """
        nx, ny = self.conf["checkerboard"]["num_inner_corners"]
        ret, corners = cv2.findChessboardCorners(im, (nx, ny), None)

        if ret:
            # Refine corner positions
            corners = cv2.cornerSubPix(
                im, corners, self.subpix_window_size, (-1, -1), self.term_crit
            )
        else:
            corners = None

        return ret, corners

    def read(self, image_name):
        im = cv2.imread(image_name)
        if self.conf.image.resize_to is not None:
            im = cv2.resize(im, self.conf.resize_to)
        h, w = im.shape[:2]
        self.image_size = w, h
        return im, w, h

    def accumulate_points(self):
        """Collect data for the optimization problem that will find the
        intrinsic camera parameters.
        """
        failures = []
        for name in self.image_names:
            im, h, w = self.read(name)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, px_corners = self.get_corner_points(im_gray)

            if ret is False:
                print(f"Skipping {name} - corners not found")
                failures.append(name)
            else:
                print(f"{name} - corners found")
                self.world_point_list.append(self.world_corners)
                self.image_point_list.append(px_corners)

        for name in failures:
            self.image_names.remove(name)

    def solve(self):
        results = cv2.calibrateCamera(
            self.world_point_list, self.image_point_list, self.image_size, None, None
        )
        ret, self.K, self.dist_coeffs, self.rvecs, self.tvecs = results
        return ret

    def undistort(self, points):
        # The OpenCV undistort function returns points in normalized image
        # coordinates, i.e. points referenced from the principal point near the
        # image center, and scaled as a fraction of the image height and width.
        u_n = cv2.undistortPoints(points, self.K, self.dist_coeffs).squeeze()

        # To convert to pixel coordinates, scale by fx, fy and offset by cx, cy.
        # In other words, multiply by the camera matrix.
        u = ic(self.K @ hc(u_n))  # (npoints, 2)
        return u

    def report(self):
        # Distortion parameters [k1, k2, p1, p2, k3]
        # k1, k2, k3: radial distortion coefficients where
        #             [x,y]_corrected = [x,y]*(1 + k1 r^2 + k2 r^4 + k3 r^6)
        # p1, p2: tangential distortion coefficients where
        #             x_corrected = x + (2p1xy + p2(r^2 + 2x^2))
        #             y_corrected = y + (p1(r^2 + 2y^2) + 2p2xy)
        w, h = self.image_size
        print("\nImage size (w x h):\n", w, h)
        print("\nDistortion params:\n", self.dist_coeffs)
        print("\nCamera matrix:\n", self.K)

        # Coeffs of image width and height are invariant to image resizing.
        # If the sensor w, h are known, f and d can be expressed in physical
        # units.
        fx, fy = self.K[0][0], self.K[1][1]
        cx, cy = self.K[0][2], self.K[1][2]
        print()
        print("fx = {:.3f}*w".format(fx / w))
        print("fy = {:.3f}*h".format(fy / h))
        print("cx = {:.3f}*w".format(cx / w))
        print("cy = {:.3f}*h".format(cy / h))

        print(f"\nMean reprojection error: {self.mean_reprojection_error():.4f}")

    def mean_reprojection_error(self):
        mean_error = 0
        w, h = self.image_size
        for i, _ in enumerate(self.world_point_list):
            im_pts, jac = cv2.projectPoints(
                self.world_point_list[i],
                self.rvecs[i],
                self.tvecs[i],
                self.K,
                self.dist_coeffs,
            )
            n = len(im_pts)
            error = cv2.norm(self.image_point_list[i], im_pts, cv2.NORM_L2) / n
            mean_error += error
        # mean_error /= len(self.world_point_list) * np.hypot(w, h)
        return mean_error

    def run(self):
        self.accumulate_points()
        self.solve()
        self.report()

    def save_corrected_images(self, output_dir: Path = Path("/tmp")):
        for i, (name, pts) in enumerate(zip(self.image_names, self.image_point_list)):
            im, w, h = self.read(name)
            im_undist = cv2.undistort(im, self.K, self.dist_coeffs)
            upts = self.undistort(pts)

            # Original corner points in magenta
            for p in pts:
                x, y = p.ravel() + 0.5
                cv2.circle(im_undist, (int(x), int(y)), 4, (255, 0, 255), 2)

            # Corrected points in green
            for p in upts:
                x, y = p.ravel() + 0.5
                cv2.circle(im_undist, (int(x), int(y)), 4, (0, 255, 0), 2)

            outfile = output_dir / Path(name).name
            print(f"saving {outfile}")
            cv2.imwrite(str(outfile), im_undist)
