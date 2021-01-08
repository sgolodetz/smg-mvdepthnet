import cv2
import numpy as np

from typing import Optional

from smg.mvdepthnet.mvdepthestimator import MVDepthEstimator
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker


def main() -> None:
    with OpenNICamera(mirror_images=True) as camera:
        with RGBDTracker(
                settings_file=f"settings-kinect.yaml", use_viewer=True,
                voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            fx, fy, cx, cy = camera.get_colour_intrinsics()
            model_path: str = "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"

            intrinsics: np.ndarray = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

            estimator: MVDepthEstimator = MVDepthEstimator(model_path, intrinsics)

            reference_image: Optional[np.ndarray] = None
            reference_pose: Optional[np.ndarray] = None
            estimated_depth_image: Optional[np.ndarray] = None

            while True:
                colour_image, depth_image = camera.get_images()
                cv2.imshow("Colour Image", colour_image)
                cv2.imshow("Depth Image", depth_image / 2)
                c: int = cv2.waitKey(1)

                if not tracker.is_ready():
                    continue

                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image, depth_image)
                if pose is None:
                    continue

                if c == ord('r'):
                    reference_image = colour_image.copy()
                    reference_pose = pose.copy()
                    continue

                if c == ord('v'):
                    break

                if reference_image is not None:
                    estimated_depth_image = estimator.estimate_depth(
                        colour_image, reference_image, np.linalg.inv(pose), np.linalg.inv(reference_pose)
                    )
                    cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)
                    cv2.waitKey(1)

            if estimated_depth_image is not None:
                estimated_depth_image = cv2.resize(estimated_depth_image, (640, 480), interpolation=cv2.INTER_NEAREST)
                VisualisationUtil.visualise_rgbd_image(
                    colour_image, estimated_depth_image, camera.get_colour_intrinsics()
                )


if __name__ == "__main__":
    main()
