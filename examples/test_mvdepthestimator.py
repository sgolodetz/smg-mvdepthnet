import cv2
import numpy as np

from smg.mvdepthnet.MVDepthEstimator import MVDepthEstimator


def main():
    model_path: str = "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"

    intrinsics: np.ndarray = np.array([
        [504.261, 0, 352.457],
        [0, 503.905, 272.202],
        [0, 0, 1]
    ])

    estimator: MVDepthEstimator = MVDepthEstimator(model_path, intrinsics)

    left_image: np.ndarray = cv2.imread("C:/spaint/build/bin/apps/spaintgui/sequences/Teddy/frame-000000.color.png")
    right_image: np.ndarray = cv2.imread("C:/spaint/build/bin/apps/spaintgui/sequences/Teddy/frame-000050.color.png")
    left_pose: np.ndarray = np.eye(4)
    right_pose: np.ndarray = np.array([
        [0.994802, 0.0166392, 0.100461, -0.0844009],
        [-0.0250872, 0.996198, 0.083425, -0.0346446],
        [-0.0986906, -0.0855116, 0.991437, 0.0318293],
        [0, 0, -0, 1]
    ])

    depth_image: np.ndarray = estimator.estimate_depth(left_image, right_image, left_pose, right_pose)

    # FIXME: Use pyplot instead.
    cv2.imshow("Depth Image", depth_image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
