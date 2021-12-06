from __future__ import annotations

import cv2
import numpy as np

from operator import itemgetter
from typing import List, Optional, Tuple

from smg.rigging.helpers import CameraUtil
from smg.utility import DepthImageProcessor, GeometryUtil, ImageUtil, MonocularDepthEstimator

from .mvdepth_multiview_depth_estimator import MVDepthMultiviewDepthEstimator


class MVDepthMonocularDepthEstimator(MonocularDepthEstimator):
    """A monocular depth estimator based on MVDepthNet that triangulates against a single keyframe."""

    # CONSTRUCTOR

    def __init__(self, model_path: str, *, border_to_fill: int = 40, debug: bool = False,
                 max_depth: float = 3.0, max_rotation_before_keyframe: float = 5.0,
                 max_rotation_for_triangulation: float = 20.0, max_translation_before_keyframe: float = 0.05,
                 min_translation_for_triangulation: float = 0.025):
        """
        Construct a monocular depth estimator based on MVDepthNet that triangulates against a single keyframe.

        :param model_path:                          The path to the MVDepthNet model.
        :param border_to_fill:                      The size of the border (in pixels) of the estimated depth image
                                                    that is to be filled with zeros to help mitigate depth noise.
        :param debug:                               Whether to show debug visualisations.
        :param max_translation_before_keyframe:     The maximum translation (in m) there can be between the current
                                                    position and the position of the closest keyframe without
                                                    triggering the creation of a new keyframe.
        :param max_depth:                           The maximum depth values to keep during post-processing (pixels with
                                                    depth values greater than this will have their depths set to zero).
        :param max_rotation_before_keyframe:        The maximum rotation (in degrees) there can be between the current
                                                    look vector and the look vector of the closest keyframe without
                                                    triggering the creation of a new keyframe.
        :param max_rotation_for_triangulation:      The maximum rotation (in degrees) there can be between the look
                                                    vector of a keyframe and the current look vector for the keyframe
                                                    to be used.
        :param min_translation_for_triangulation:   The minimum translation (in m) there can be between the position
                                                    of a keyframe and the current position for the keyframe to be used.
        """
        self.__border_to_fill: int = border_to_fill
        self.__debug: bool = debug
        self.__intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.__keyframes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.__max_depth: float = max_depth
        self.__max_rotation_before_keyframe: float = max_rotation_before_keyframe
        self.__max_rotation_for_triangulation: float = max_rotation_for_triangulation
        self.__max_translation_before_keyframe: float = max_translation_before_keyframe
        self.__min_translation_for_triangulation: float = min_translation_for_triangulation
        self.__multiview_depth_estimator: MVDepthMultiviewDepthEstimator = MVDepthMultiviewDepthEstimator(model_path)

        # Additional variables needed to perform temporal filtering of the depth images.
        self.__current_depth_image: Optional[np.ndarray] = None
        self.__current_w_t_c: Optional[np.ndarray] = None
        self.__current_world_points: Optional[np.ndarray] = None
        self.__previous_depth_image: Optional[np.ndarray] = None
        self.__previous_w_t_c: Optional[np.ndarray] = None
        self.__previous_world_points: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def estimate_depth(self, colour_image: np.ndarray, tracker_w_t_c: np.ndarray, *, postprocess: bool = False) \
            -> Optional[np.ndarray]:
        """
        Try to estimate a depth image corresponding to the colour image passed in.

        .. note::
            If a suitable keyframe cannot be found for triangulation, this will return None.

        :param colour_image:    The colour image.
        :param tracker_w_t_c:   The camera pose corresponding to the colour image (as a camera -> world transform).
        :param postprocess:     Whether or not to apply any optional post-processing to the depth image.
        :return:                The estimated depth image, if possible, or None otherwise.
        """
        estimated_depth_image: Optional[np.ndarray] = None

        # Compute the translations (in m) and (look) rotations (in degrees) with respect to any existing keyframes.
        translations: List[float] = []
        rotations: List[float] = []
        for _, keyframe_w_t_c in self.__keyframes:
            translations.append(CameraUtil.compute_translation_p(tracker_w_t_c, keyframe_w_t_c))
            rotations.append(CameraUtil.compute_look_rotation_p(tracker_w_t_c, keyframe_w_t_c))

        # Score all of the keyframes with respect to the current frame.
        scores: List[(int, float)] = []
        translation_to_closest_keyframe: float = np.inf
        rotation_to_closest_keyframe: float = np.inf

        for i in range(len(self.__keyframes)):
            if translations[i] < translation_to_closest_keyframe:
                translation_to_closest_keyframe = translations[i]
                rotation_to_closest_keyframe = rotations[i]

            if translations[i] < self.__min_translation_for_triangulation \
                    or rotations[i] > self.__max_rotation_for_triangulation:
                # If the translation's too small, or the rotation's too large, force the score of this keyframe to 0.
                scores.append((i, 0.0))
            else:
                # Otherwise, compute a score loosely based on the one in the Mobile3DRecon paper. Note that we don't
                # use the rotation part of the score, as it produces bad results, and we change the parameters for
                # the translation part of the score (as these parameters empirically seem to work better).
                b_m: float = 0.4
                delta: float = 0.2
                w_b: float = np.exp(-(translations[i] - b_m) ** 2 / delta ** 2)
                scores.append((i, w_b))

        # Try to choose a keyframe to use together with the current frame to estimate the depth.
        if len(scores) >= 1:
            # Find the best keyframe, based on the scores.
            # FIXME: There's no need to fully sort the list here.
            # See: https://stackoverflow.com/a/23734295/499449
            # x[np.argpartition(x, range(-2,0))[::-1]
            scores = sorted(scores, key=itemgetter(1), reverse=True)
            best_keyframe_idx, best_keyframe_score = scores[0]

            # If the best keyframe is fine to use:
            if best_keyframe_score > 0.0:
                # Look up the keyframe image and pose.
                best_keyframe_image, best_keyframe_w_t_c = self.__keyframes[best_keyframe_idx]

                # Estimate the depth image.
                estimated_depth_image = self.__multiview_depth_estimator.estimate_depth(
                    colour_image, best_keyframe_image, tracker_w_t_c, best_keyframe_w_t_c
                )

        # Check whether this frame should be a new keyframe. If so, add it to the list.
        if translation_to_closest_keyframe > self.__max_translation_before_keyframe \
                or rotation_to_closest_keyframe > self.__max_rotation_before_keyframe:
            self.__keyframes.append((colour_image.copy(), tracker_w_t_c.copy()))

        # If a depth image was successfully estimated:
        if estimated_depth_image is not None:
            # Fill the border with zeros (depths around the image border are often quite noisy).
            estimated_depth_image = ImageUtil.fill_border(estimated_depth_image, self.__border_to_fill, 0.0)

            # If we're debugging, show the estimated depth image.
            if self.__debug:
                cv2.imshow("Raw Estimated Depth Image", estimated_depth_image / 5)
                cv2.waitKey(1)

            # Return the estimated depth image, fully post-processing it in the process if requested,
            # or just limiting the maximum depth as requested otherwise.
            if postprocess:
                return self.__postprocess_depth_image(estimated_depth_image, tracker_w_t_c)
            else:
                return np.where(estimated_depth_image <= self.__max_depth, estimated_depth_image, 0.0)
        else:
            return None

    def get_keyframes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the current set of keyframes.

        :return:    The current set of keyframes.
        """
        return self.__keyframes

    def set_intrinsics(self, intrinsics: np.ndarray) -> MonocularDepthEstimator:
        """
        Set the camera intrinsics.

        :param intrinsics:  The 3x3 camera intrinsics matrix.
        :return:            The current object.
        """
        self.__intrinsics = GeometryUtil.intrinsics_to_tuple(intrinsics)
        self.__multiview_depth_estimator.set_intrinsics(intrinsics)
        return self

    # PRIVATE METHODS

    def __postprocess_depth_image(self, depth_image: np.ndarray, tracker_w_t_c: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to post-process the specified depth image to reduce the amount of noise it contains.

        .. note::
            This function will return None if the input depth image does not have depth values for enough pixels.

        :param depth_image:     The input depth image.
        :param tracker_w_t_c:   The camera pose corresponding to the depth image (as a camera -> world transform).
        :return:                The post-processed depth image, if possible, or None otherwise.
        """
        # FIXME: This is all but identical to the one in DVMVSMonocularDepthEstimator - unify them.
        # Update the variables needed to perform temporal filtering.
        self.__previous_depth_image = self.__current_depth_image
        self.__previous_w_t_c = self.__current_w_t_c
        self.__previous_world_points = self.__current_world_points

        self.__current_depth_image = depth_image.copy()
        self.__current_w_t_c = tracker_w_t_c.copy()
        self.__current_world_points = GeometryUtil.compute_world_points_image_fast(
            depth_image, tracker_w_t_c, self.__intrinsics
        )

        # If a previous depth image is available with which to perform temporal filtering:
        if self.__previous_depth_image is not None:
            # Remove any pixel in the current depth image that either does not have a corresponding pixel
            # in the previous world-space points image at all, or else does not have one whose world-space
            # point is sufficiently close to the current world-space point.
            postprocessed_depth_image = DepthImageProcessor.remove_temporal_inconsistencies(
                self.__current_depth_image, self.__current_w_t_c, self.__current_world_points,
                self.__previous_depth_image, self.__previous_w_t_c, self.__previous_world_points,
                self.__intrinsics, debug=False, distance_threshold=0.1
            )

            # Further post-process the resulting depth image using the normal approach, and then return it.
            return DepthImageProcessor.postprocess_depth_image(
                postprocessed_depth_image, max_depth=self.__max_depth, max_depth_difference=0.05,
                median_filter_radius=5, min_region_size=5000, min_valid_fraction=0.0
            )

        # Otherwise, skip this depth image and wait till next time.
        else:
            return None
