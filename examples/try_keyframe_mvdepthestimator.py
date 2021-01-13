import cv2
import numpy as np

from operator import itemgetter
from typing import List, Optional, Tuple

from smg.mvdepthnet.mvdepthestimator import MVDepthEstimator
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.rigging.helpers import CameraUtil
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
            depth_estimator: MVDepthEstimator = MVDepthEstimator(
                "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar",
                GeometryUtil.intrinsics_to_matrix(camera.get_colour_intrinsics())
            )

            # Initialise the list of keyframes.
            keyframes: List[Tuple[np.ndarray, np.ndarray]] = []

            best_depth_image: Optional[np.ndarray] = None
            second_best_depth_image: Optional[np.ndarray] = None

            while True:
                # Get the colour and depth images from the camera, and show them.
                colour_image, depth_image = camera.get_images()
                cv2.imshow("Colour Image", colour_image)
                cv2.imshow("Depth Image", depth_image / 2)
                c: int = cv2.waitKey(1)

                # If the tracker's not yet ready, or the pose can't be estimated for this frame, continue.
                if not tracker.is_ready():
                    continue

                tracker_c_t_w: Optional[np.ndarray] = tracker.estimate_pose(colour_image, depth_image)
                if tracker_c_t_w is None:
                    continue

                tracker_w_t_c: np.ndarray = np.linalg.inv(tracker_c_t_w)

                # If the user presses the 'v' key, exit the loop so that the estimated depth image can be visualised.
                if c == ord('v'):
                    break

                # Compute the baselines (in m) and look angles (in degrees) with respect to any existing keyframes.
                baselines: List[float] = []
                look_angles: List[float] = []
                for _, keyframe_w_t_c in keyframes:
                    baselines.append(CameraUtil.compute_baseline_p(tracker_w_t_c, keyframe_w_t_c))
                    look_angles.append(CameraUtil.compute_look_angle_p(tracker_w_t_c, keyframe_w_t_c))

                # Score all of the keyframes with respect to the current frame.
                scores: List[(int, float)] = []
                smallest_baseline: float = 1000.0
                smallest_look_angle: float = 1000.0

                for i in range(len(keyframes)):
                    smallest_baseline = min(baselines[i], smallest_baseline)
                    smallest_look_angle = min(look_angles[i], smallest_look_angle)

                    if baselines[i] < 0.025 or look_angles[i] > 20.0:
                        scores.append((i, 0.0))
                    else:
                        # Adapted from the Mobile3DRecon paper.
                        b_m: float = 0.15
                        delta: float = 0.1
                        alpha_m: float = 10.0
                        w_b: float = np.exp(-(baselines[i] - b_m) ** 2 / delta ** 2)
                        w_v: float = max(alpha_m / look_angles[i], 1)
                        scores.append((i, w_b * w_v))

                # Try to choose up to two keyframes to use together with the current frame to estimate the depth.
                if len(scores) >= 2:
                    # FIXME: There's no need to fully sort the list here.
                    scores = sorted(scores, key=itemgetter(1), reverse=True)
                    best_keyframe_idx, best_keyframe_score = scores[0]
                    second_best_keyframe_idx, second_best_keyframe_score = scores[1]
                    if best_keyframe_score > 0.0 and second_best_keyframe_score > 0.0:
                        best_keyframe_image, best_keyframe_w_t_c = keyframes[best_keyframe_idx]
                        second_best_keyframe_image, second_best_keyframe_w_t_c = keyframes[second_best_keyframe_idx]

                        best_depth_image = depth_estimator.estimate_depth(
                            colour_image, best_keyframe_image, tracker_w_t_c, best_keyframe_w_t_c
                        )
                        second_best_depth_image = depth_estimator.estimate_depth(
                            colour_image, second_best_keyframe_image, tracker_w_t_c, second_best_keyframe_w_t_c
                        )

                        diff = np.abs(best_depth_image - second_best_depth_image)
                        best_depth_image = np.where(diff < 0.1, best_depth_image, 0.0)
                        second_best_depth_image = np.where(diff < 0.1, second_best_depth_image, 0.0)

                        cv2.imshow("Best Depth Image", best_depth_image / 2)
                        cv2.imshow("Second Best Depth Image", second_best_depth_image / 2)
                        cv2.waitKey(1)

                # Check whether this frame should be a new keyframe. If so, add it to the list.
                if smallest_baseline > 0.05 or smallest_look_angle > 5.0:
                    keyframes.append((colour_image, tracker_w_t_c))

            # Visualise the 3D point cloud corresponding to the most recently estimated best depth image (if any).
            if best_depth_image is not None:
                VisualisationUtil.visualise_rgbd_image(
                    colour_image, best_depth_image, camera.get_colour_intrinsics()
                )


if __name__ == "__main__":
    main()
