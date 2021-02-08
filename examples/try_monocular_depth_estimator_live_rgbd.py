import cv2
import numpy as np

from typing import Optional

from smg.mvdepthnet import MonocularDepthEstimator
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.utility import GeometryUtil


def main() -> None:
    # Construct the camera.
    with OpenNICamera(mirror_images=True) as camera:
        # Construct the tracker.
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=False,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            # Construct the depth estimator.
            depth_estimator: MonocularDepthEstimator = MonocularDepthEstimator(
                "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar", debug=True
            ).set_intrinsics(GeometryUtil.intrinsics_to_matrix(camera.get_colour_intrinsics()))

            # noinspection PyUnusedLocal
            estimated_depth_image: Optional[np.ndarray] = None

            # Repeatedly:
            while True:
                # Get the colour and depth images from the camera, and show them.
                colour_image, depth_image = camera.get_images()
                cv2.imshow("Colour Image", colour_image)
                cv2.imshow("Depth Image", depth_image / 2)
                c: int = cv2.waitKey(1)

                # If the user presses the 'v' key, exit the loop so that any estimated depth image can be visualised.
                if c == ord('v'):
                    break

                # If the tracker's not yet ready, or the pose can't be estimated for this frame, continue.
                if not tracker.is_ready():
                    continue

                tracker_c_t_w: Optional[np.ndarray] = tracker.estimate_pose(colour_image, depth_image)
                if tracker_c_t_w is None:
                    continue

                tracker_w_t_c: np.ndarray = np.linalg.inv(tracker_c_t_w)

                # Estimate the depth image.
                estimated_depth_image = depth_estimator.estimate_depth(colour_image, tracker_w_t_c)

            # Visualise the 3D point cloud corresponding to the most recently estimated depth image (if any).
            if estimated_depth_image is not None:
                VisualisationUtil.visualise_rgbd_image(
                    colour_image, estimated_depth_image, camera.get_colour_intrinsics()
                )


if __name__ == "__main__":
    main()
