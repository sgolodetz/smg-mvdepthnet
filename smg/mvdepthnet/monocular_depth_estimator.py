import cv2
import numpy as np

from operator import itemgetter
from typing import List, Optional, Tuple

from smg.rigging.helpers import CameraUtil

from .multiview_depth_estimator import MultiviewDepthEstimator


class MonocularDepthEstimator:
    """A monocular depth estimator based on MVDepthNet."""

    # CONSTRUCTOR

    def __init__(self, model_path: str, intrinsics: np.ndarray, *, debug: bool = False):
        """
        Construct a monocular depth estimator.

        :param model_path:  The path to the MVDepthNet model.
        :param intrinsics:  The 3x3 camera intrinsics matrix.
        :param debug:       Whether to show debug visualisations.
        """
        self.__debug: bool = debug
        self.__keyframes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.__multiview_depth_estimator: MultiviewDepthEstimator = MultiviewDepthEstimator(model_path, intrinsics)

    # PUBLIC METHODS

    def estimate_depth(self, colour_image: np.ndarray, tracker_w_t_c: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to estimate a depth image corresponding to the colour image passed in.

        .. note::
            If two suitable keyframes cannot be found for triangulation, this will return None.

        :param colour_image:    The colour image.
        :param tracker_w_t_c:   The camera pose corresponding to the colour image (as a camera -> world transform).
        :return:                The estimated depth image, if possible, or None otherwise.
        """
        best_depth_image: Optional[np.ndarray] = None

        # Compute the baselines (in m) and look angles (in degrees) with respect to any existing keyframes.
        baselines: List[float] = []
        look_angles: List[float] = []
        for _, keyframe_w_t_c in self.__keyframes:
            baselines.append(CameraUtil.compute_baseline_p(tracker_w_t_c, keyframe_w_t_c))
            look_angles.append(CameraUtil.compute_look_angle_p(tracker_w_t_c, keyframe_w_t_c))

        # Score all of the keyframes with respect to the current frame.
        scores: List[(int, float)] = []
        smallest_baseline: float = 1000.0
        smallest_look_angle: float = 1000.0

        for i in range(len(self.__keyframes)):
            smallest_baseline = min(baselines[i], smallest_baseline)
            smallest_look_angle = min(look_angles[i], smallest_look_angle)

            if baselines[i] < 0.025 or look_angles[i] > 20.0:
                # If the baseline's too small, force the score of this keyframe to 0.
                scores.append((i, 0.0))
            else:
                # Otherwise, compute a score as per the Mobile3DRecon paper (but with different parameters).
                b_m: float = 0.15
                delta: float = 0.1
                alpha_m: float = 10.0
                w_b: float = np.exp(-(baselines[i] - b_m) ** 2 / delta ** 2)
                w_v: float = max(alpha_m / look_angles[i], 1)
                scores.append((i, w_b * w_v))

        # Try to choose up to two keyframes to use together with the current frame to estimate the depth.
        if len(scores) >= 2:
            # Find the two best keyframes, based on their scores.
            # FIXME: There's no need to fully sort the list here.
            scores = sorted(scores, key=itemgetter(1), reverse=True)
            best_keyframe_idx, best_keyframe_score = scores[0]
            second_best_keyframe_idx, second_best_keyframe_score = scores[1]

            # If both keyframes are fine to use:
            if best_keyframe_score > 0.0 and second_best_keyframe_score > 0.0:
                # Look up the keyframe images and poses.
                best_keyframe_image, best_keyframe_w_t_c = self.__keyframes[best_keyframe_idx]
                second_best_keyframe_image, second_best_keyframe_w_t_c = self.__keyframes[second_best_keyframe_idx]

                # Separately estimate a depth image from each keyframe.
                best_depth_image = self.__multiview_depth_estimator.estimate_depth(
                    colour_image, best_keyframe_image, tracker_w_t_c, best_keyframe_w_t_c
                )
                second_best_depth_image: np.ndarray = self.__multiview_depth_estimator.estimate_depth(
                    colour_image, second_best_keyframe_image, tracker_w_t_c, second_best_keyframe_w_t_c
                )

                # Filter out any depths that are not sufficiently consistent across both estimates.
                tolerance: float = 0.1
                diff: np.ndarray = np.abs(best_depth_image - second_best_depth_image)
                best_depth_image = np.where(diff < tolerance, best_depth_image, 0.0)

                # If we're debugging, also filter the second-best depth image, and show both depth images.
                if self.__debug:
                    second_best_depth_image = np.where(diff < tolerance, second_best_depth_image, 0.0)
                    cv2.imshow("Best Depth Image", best_depth_image / 2)
                    cv2.imshow("Second Best Depth Image", second_best_depth_image / 2)
                    cv2.waitKey(1)

        # Check whether this frame should be a new keyframe. If so, add it to the list.
        if smallest_baseline > 0.05 or smallest_look_angle > 5.0:
            self.__keyframes.append((colour_image.copy(), tracker_w_t_c.copy()))

        return best_depth_image
