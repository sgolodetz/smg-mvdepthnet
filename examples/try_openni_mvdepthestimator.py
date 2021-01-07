import cv2
import numpy as np

from typing import Optional

from smg.mvdepthnet.mvdepthestimator import MVDepthEstimator
from smg.openni import OpenNICamera
from smg.pyorbslam2 import MonocularTracker


def main() -> None:
    with OpenNICamera(mirror_images=True) as camera:
        with MonocularTracker(
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

            while True:
                colour_image, depth_image = camera.get_images()
                cv2.imshow("Colour Image", colour_image)
                cv2.imshow("Depth Image", depth_image / 2)
                c: int = cv2.waitKey(1)

                if not tracker.is_ready():
                    continue

                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)  #, depth_image)
                if pose is None:
                    continue

                if c == ord('r'):
                    reference_image = colour_image.copy()
                    reference_pose = pose.copy()
                    continue

                if reference_image is not None:
                    estimated_depth_image: np.ndarray = estimator.estimate_depth(
                        colour_image, reference_image, np.linalg.inv(pose), np.linalg.inv(reference_pose)
                    )
                    cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)
                    cv2.waitKey(1)

        # estimator: MVDepthEstimator = MVDepthEstimator(model_path, intrinsics)
        #
        # left_image: np.ndarray = cv2.imread("D:/7scenes/heads/train/frame-000000.color.png")
        # right_image: np.ndarray = cv2.imread("D:/7scenes/heads/train/frame-000055.color.png")
        # left_pose: np.ndarray = np.array([
        #     [9.9798417e-001, 2.5588147e-002, 5.5003788e-002, 9.5729552e-002],
        #     [-3.4286145e-002, 9.8580331e-001, 1.6358595e-001, -4.1082110e-002],
        #     [-5.0044306e-002, -1.6512756e-001, 9.8484284e-001, 1.3248709e-001],
        #     [0.0000000e+000, 0.0000000e+000, 0.0000000e+000, 1.0000000e+000]
        # ])
        # right_pose: np.ndarray = np.array([
        #     [9.9853259e-001, -4.9765650e-002, 9.9766739e-003, 2.9618734e-001],
        #     [4.7499087e-002, 9.8555100e-001, 1.6177388e-001, -1.2100209e-001],
        #     [-1.7887015e-002, -1.6104801e-001, 9.8662180e-001, -1.5921636e-002],
        #     [0.0000000e+000, 0.0000000e+000, 0.0000000e+000, 1.0000000e+000]
        # ])
        #
        # print(left_pose)
        # print(right_pose)
        #
        # depth_image: np.ndarray = estimator.estimate_depth(left_image, right_image, left_pose, right_pose)
        #
        # plt.imshow(depth_image)
        # plt.show()
        # pass


if __name__ == "__main__":
    main()
