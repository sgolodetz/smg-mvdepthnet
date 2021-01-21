from __future__ import annotations

import cv2
import numpy as np

from operator import itemgetter
from typing import List, Optional, Tuple

from smg.rigging.helpers import CameraUtil

from .multiview_depth_estimator import MultiviewDepthEstimator


class MonocularDepthEstimator:
    """A monocular depth estimator based on MVDepthNet."""

    # CONSTRUCTOR

    def __init__(self, model_path: str, *, debug: bool = False, max_consistent_depth_diff: float = 0.1,
                 max_rotation_before_keyframe: float = 5.0, max_rotation_for_triangulation: float = 20.0,
                 max_translation_before_keyframe: float = 0.05, min_translation_for_triangulation: float = 0.025):
        """
        Construct a monocular depth estimator.

        :param model_path:                          The path to the MVDepthNet model.
        :param debug:                               Whether to show debug visualisations.
        :param max_translation_before_keyframe:     The maximum translation (in m) there can be between the current
                                                    position and the position of the closest keyframe without
                                                    triggering the creation of a new keyframe.
        :param max_consistent_depth_diff:           The maximum difference there can be between the depths estimated
                                                    for a pixel by the best and second best keyframes for those depths
                                                    to be considered sufficiently consistent.
        :param max_rotation_before_keyframe:        The maximum rotation (in degrees) there can be between the current
                                                    look vector and the look vector of the closest keyframe without
                                                    triggering the creation of a new keyframe.
        :param max_rotation_for_triangulation:      The maximum rotation (in degrees) there can be between the look
                                                    vector of a keyframe and the current look vector for the keyframe
                                                    to be used.
        :param min_translation_for_triangulation:   The minimum translation (in m) there can be between the position
                                                    of a keyframe and the current position for the keyframe to be used.
        """
        self.__debug: bool = debug
        self.__keyframes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.__max_consistent_depth_diff: float = max_consistent_depth_diff
        self.__max_rotation_before_keyframe: float = max_rotation_before_keyframe
        self.__max_rotation_for_triangulation: float = max_rotation_for_triangulation
        self.__max_translation_before_keyframe: float = max_translation_before_keyframe
        self.__min_translation_for_triangulation: float = min_translation_for_triangulation
        self.__multiview_depth_estimator: MultiviewDepthEstimator = MultiviewDepthEstimator(model_path)

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

        # Compute the translations (in m) and (look) rotations (in degrees) with respect to any existing keyframes.
        translations: List[float] = []
        rotations: List[float] = []
        for _, keyframe_w_t_c in self.__keyframes:
            translations.append(CameraUtil.compute_translation_p(tracker_w_t_c, keyframe_w_t_c))
            rotations.append(CameraUtil.compute_look_rotation_p(tracker_w_t_c, keyframe_w_t_c))

        # Score all of the keyframes with respect to the current frame.
        scores: List[(int, float)] = []
        smallest_translation: float = np.inf
        smallest_rotation: float = np.inf

        for i in range(len(self.__keyframes)):
            smallest_translation = min(translations[i], smallest_translation)
            smallest_rotation = min(rotations[i], smallest_rotation)

            if translations[i] < self.__min_translation_for_triangulation \
                    or rotations[i] > self.__max_rotation_for_triangulation:
                # If the translation's too small, or the rotation's too large, force the score of this keyframe to 0.
                scores.append((i, 0.0))
            else:
                # Otherwise, compute a score as per the Mobile3DRecon paper (but with different parameters).
                b_m: float = 0.15
                delta: float = 0.1
                alpha_m: float = 10.0
                w_b: float = np.exp(-(translations[i] - b_m) ** 2 / delta ** 2)
                w_v: float = max(alpha_m / rotations[i], 1)
                scores.append((i, w_b * w_v))

        # Try to choose up to two keyframes to use together with the current frame to estimate the depth.
        if len(scores) >= 2:
            # Find the two best keyframes, based on their scores.
            # FIXME: There's no need to fully sort the list here.
            # See: https://stackoverflow.com/a/23734295/499449
            # x[np.argpartition(x, range(-2,0))[::-1]
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
                diff: np.ndarray = np.abs(best_depth_image - second_best_depth_image)
                best_depth_image = np.where(diff < self.__max_consistent_depth_diff, best_depth_image, 0.0)

                # If we're debugging, also filter the second-best depth image, and show both depth images.
                if self.__debug:
                    second_best_depth_image = np.where(
                        diff < self.__max_consistent_depth_diff, second_best_depth_image, 0.0
                    )
                    cv2.imshow("Best Depth Image", best_depth_image / 2)
                    cv2.imshow("Second Best Depth Image", second_best_depth_image / 2)
                    cv2.waitKey(1)

        # Check whether this frame should be a new keyframe. If so, add it to the list.
        if smallest_translation > self.__max_translation_before_keyframe \
                or smallest_rotation > self.__max_rotation_before_keyframe:
            self.__keyframes.append((colour_image.copy(), tracker_w_t_c.copy()))

        return best_depth_image

    def set_intrinsics(self, intrinsics: np.ndarray) -> MonocularDepthEstimator:
        """
        Set the camera intrinsics.

        :param intrinsics:  The 3x3 camera intrinsics matrix.
        :return:            The current object.
        """
        self.__multiview_depth_estimator.set_intrinsics(intrinsics)
        return self
