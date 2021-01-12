import cv2
import numpy as np

from typing import List, Optional, Tuple

from smg.mvdepthnet.mvdepthestimator import MVDepthEstimator
from smg.openni import OpenNICamera
from smg.pyorbslam2 import RGBDTracker
from smg.rigging.helpers import CameraUtil
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
            depth_estimator: MVDepthEstimator = MVDepthEstimator(
                "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar",
                GeometryUtil.intrinsics_to_matrix(camera.get_colour_intrinsics())
            )

            # Initialise the list of keyframes.
            keyframes: List[Tuple[np.ndarray, np.ndarray]] = []

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

                # Compute the baselines (in m) and look angles (in degrees) with respect to any existing keyframes.
                baselines: List[float] = []
                look_angles: List[float] = []
                for _, keyframe_w_t_c in keyframes:
                    baselines.append(CameraUtil.compute_baseline_p(tracker_w_t_c, keyframe_w_t_c))
                    look_angles.append(CameraUtil.compute_look_angle_p(tracker_w_t_c, keyframe_w_t_c))

                # Try to choose a keyframe to use together with the current frame to estimate the depth.
                best_keyframe_idx: int = -1
                best_score: float = 0.0
                smallest_baseline: float = 1000.0
                smallest_look_angle: float = 1000.0
                print(f"Keyframe Count: {len(keyframes)}")
                for i in range(len(keyframes)):
                    smallest_baseline = min(baselines[i], smallest_baseline)
                    smallest_look_angle = min(look_angles[i], smallest_look_angle)

                    if not (0.1 <= baselines[i] <= 0.5):
                        continue
                    if not (5.0 <= look_angles[i] <= 15.0):
                        continue

                    # See the Mobile3DRecon paper.
                    b_m: float = 0.3
                    delta: float = 0.2
                    alpha_m: float = 10.0
                    w_b: float = np.exp(-(baselines[i] - b_m)**2 / delta**2)
                    w_v: float = max(alpha_m / look_angles[i], 1)
                    score: float = w_b * w_v
                    if score > best_score:
                        best_keyframe_idx = i
                        best_score = score

                force_new_keyframe: bool = False

                if best_keyframe_idx >= 0:
                    # Estimate a depth image for the current frame, and show it.
                    keyframe_image, keyframe_w_t_c = keyframes[best_keyframe_idx]
                    estimated_depth_image: np.ndarray = depth_estimator.estimate_depth(
                        colour_image, keyframe_image, tracker_w_t_c, keyframe_w_t_c
                    )
                    cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)
                    cv2.waitKey(1)
                elif smallest_baseline > 0.5 or smallest_look_angle > 15.0:
                    force_new_keyframe = True

                # Check whether this frame should be a new keyframe. If so, add it to the list.
                new_keyframe: bool = True
                if not force_new_keyframe:
                    for i in range(len(keyframes)):
                        if baselines[i] <= 0.1 and look_angles[i] <= 5.0:
                            new_keyframe = False
                            break

                if new_keyframe:
                    print(f"Adding New Keyframe (Forced={force_new_keyframe})")
                    keyframes.append((colour_image, tracker_w_t_c))


if __name__ == "__main__":
    main()
