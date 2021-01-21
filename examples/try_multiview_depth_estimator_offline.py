import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Tuple

from smg.mvdepthnet import MultiviewDepthEstimator
from smg.utility import GeometryUtil, PoseUtil


def main():
    # Specify the input data and camera intrinsics.
    sequence_dir: str = "D:/7scenes/heads/train"
    intrinsics: Tuple[float, float, float, float] = (585.0, 585.0, 320.0, 240.0)
    left_idx: int = 0
    right_idx: int = 55

    # Load the input images and poses.
    left_image: np.ndarray = cv2.imread(os.path.join(sequence_dir, f"frame-{left_idx:06d}.color.png"))
    right_image: np.ndarray = cv2.imread(os.path.join(sequence_dir, f"frame-{right_idx:06d}.color.png"))
    left_pose: np.ndarray = PoseUtil.load_pose(os.path.join(sequence_dir, f"frame-{left_idx:06d}.pose.txt"))
    right_pose: np.ndarray = PoseUtil.load_pose(os.path.join(sequence_dir, f"frame-{right_idx:06d}.pose.txt"))

    # Estimate the depth for the left-hand image.
    depth_estimator: MultiviewDepthEstimator = MultiviewDepthEstimator(
        "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"
    ).set_intrinsics(GeometryUtil.intrinsics_to_matrix(intrinsics))
    depth_image: np.ndarray = depth_estimator.estimate_depth(left_image, right_image, left_pose, right_pose)

    # Show the estimated depth image.
    plt.imshow(depth_image)
    plt.show()


if __name__ == "__main__":
    main()
