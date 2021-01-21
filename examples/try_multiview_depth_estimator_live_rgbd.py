import cv2
import numpy as np

from typing import Optional

from smg.mvdepthnet import MultiviewDepthEstimator
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.utility import GeometryUtil


def main() -> None:
    # Construct the camera.
    with OpenNICamera(mirror_images=True) as camera:
        # Construct the tracker.
        with RGBDTracker(
            settings_file=f"settings-kinect.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            # Construct the depth estimator.
            depth_estimator: MultiviewDepthEstimator = MultiviewDepthEstimator(
                "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"
            ).set_intrinsics(GeometryUtil.intrinsics_to_matrix(camera.get_colour_intrinsics()))

            reference_image: Optional[np.ndarray] = None
            reference_pose: Optional[np.ndarray] = None
            estimated_depth_image: Optional[np.ndarray] = None

            while True:
                # Get the colour and depth images from the camera, and show them.
                colour_image, depth_image = camera.get_images()
                cv2.imshow("Colour Image", colour_image)
                cv2.imshow("Depth Image", depth_image / 2)
                c: int = cv2.waitKey(1)

                # If the tracker's not yet ready, or the pose can't be estimated for this frame, continue.
                if not tracker.is_ready():
                    continue

                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image, depth_image)
                if pose is None:
                    continue

                # If the user presses the 'r' key, set this frame as the reference.
                if c == ord('r'):
                    reference_image = colour_image.copy()
                    reference_pose = pose.copy()
                    continue

                # If the user presses the 'v' key, exit the loop so that the estimated depth image can be visualised.
                if c == ord('v'):
                    break

                # Provided the reference frame has been set:
                if reference_image is not None:
                    # Estimate a depth image for the current frame, and show it.
                    estimated_depth_image = depth_estimator.estimate_depth(
                        colour_image, reference_image, np.linalg.inv(pose), np.linalg.inv(reference_pose)
                    )
                    cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)
                    cv2.waitKey(1)

            # Visualise the 3D point cloud corresponding to the most recently estimated depth image (if any).
            if estimated_depth_image is not None:
                VisualisationUtil.visualise_rgbd_image(
                    colour_image, estimated_depth_image, camera.get_colour_intrinsics()
                )


if __name__ == "__main__":
    main()
