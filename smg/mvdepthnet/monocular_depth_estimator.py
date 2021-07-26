from __future__ import annotations

import cv2
import numpy as np

from operator import itemgetter
from typing import List, Optional, Tuple

from smg.rigging.helpers import CameraUtil
from smg.utility import DepthImageProcessor, ImageUtil

from .multiview_depth_estimator import MultiviewDepthEstimator


class MonocularDepthEstimator:
    """A monocular depth estimator based on MVDepthNet."""

    # CONSTRUCTOR

    def __init__(self, model_path: str, *, border_to_fill: int = 40, debug: bool = False,
                 max_consistent_depth_diff: float = 0.05, max_rotation_before_keyframe: float = 5.0,
                 max_rotation_for_triangulation: float = 20.0, max_translation_before_keyframe: float = 0.05,
                 min_translation_for_triangulation: float = 0.025):
        """
        Construct a monocular depth estimator.

        :param model_path:                          The path to the MVDepthNet model.
        :param border_to_fill:                      The size of the border (in pixels) of the estimated depth image
                                                    that is to be filled with zeros to help mitigate depth noise.
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
        self.__border_to_fill: int = border_to_fill
        self.__debug: bool = debug
        self.__keyframes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.__max_consistent_depth_diff: float = max_consistent_depth_diff
        self.__max_rotation_before_keyframe: float = max_rotation_before_keyframe
        self.__max_rotation_for_triangulation: float = max_rotation_for_triangulation
        self.__max_translation_before_keyframe: float = max_translation_before_keyframe
        self.__min_translation_for_triangulation: float = min_translation_for_triangulation
        self.__multiview_depth_estimator: MultiviewDepthEstimator = MultiviewDepthEstimator(model_path)

    # PUBLIC STATIC METHODS

    @staticmethod
    def postprocess_depth_image(depth_image: np.ndarray, *, max_depth: float = 3.0, max_depth_difference: float = 0.05,
                                median_filter_radius: int = 7, min_region_size: int = 20000,
                                min_valid_fraction: float = 0.2) -> Optional[np.ndarray]:
        """
        Try to post-process the specified depth image to try to reduce the amount of noise it contains.

        .. note::
            This function will return None if the input depth image does not have depth values for enough pixels.

        :param depth_image:             The input depth image.
        :param max_depth:               The maximum depth values to keep (pixels with depth values greater than this
                                        will have their depths set to zero).
        :param max_depth_difference:    The maximum depth difference to allow between two neighbouring pixels in the
                                        same segmentation region.
        :param median_filter_radius:    The radius of the median filter to use to reduce impulsive noise at the end
                                        of the post-processing operation.
        :param min_region_size:         The minimum size of region to keep from the depth segmentation (that is,
                                        regions smaller than this will have their depths set to zero).
        :param min_valid_fraction:      The minimum fraction of pixels for which the input depth image must have
                                        depth values for the post-processing operation to succeed. (Note that we
                                        remove pixels whose depth values are greater than the specified maximum
                                        depth before performing this test.)
        :return:                        The post-processed depth image, if possible, or None otherwise.
        """
        # Limit the depth range (more distant points can be unreliable).
        depth_image = np.where(depth_image <= max_depth, depth_image, 0.0)

        # If we have depth values for more than the specified fraction of the remaining pixels:
        if np.count_nonzero(depth_image) / np.product(depth_image.shape) >= min_valid_fraction:
            # Segment the depth image into regions such that all of the pixels in each region have similar depth.
            segmentation, stats, _ = DepthImageProcessor.segment_depth_image(
                depth_image, max_depth_difference=max_depth_difference
            )

            # Remove any regions that are smaller than the specified size.
            depth_image, _ = DepthImageProcessor.remove_small_regions(
                depth_image, segmentation, stats, min_region_size=min_region_size
            )

            # Median filter the depth image to help mitigate impulsive noise.
            depth_image = cv2.medianBlur(depth_image, median_filter_radius)

            return depth_image

        # Otherwise, discard the depth image.
        else:
            return None

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
        result: Optional[Tuple[np.ndarray, np.ndarray]] = self.estimate_depth_full(colour_image, tracker_w_t_c)
        if result is not None:
            estimated_depth_image, depth_diff_image = result

            # Filter out any depths that were not sufficiently consistent across both depth estimates.
            estimated_depth_image = np.where(
                depth_diff_image < self.__max_consistent_depth_diff, estimated_depth_image, 0.0
            )

            # If we're debugging, show the output image.
            if self.__debug:
                cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)
                cv2.waitKey(1)

            return estimated_depth_image
        else:
            return None

    def estimate_depth_full(self, colour_image: np.ndarray, tracker_w_t_c: np.ndarray) \
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Try to estimate a depth image corresponding to the colour image passed in.

        .. note::
            If two suitable keyframes cannot be found for triangulation, this will return None.

        :param colour_image:    The colour image.
        :param tracker_w_t_c:   The camera pose corresponding to the colour image (as a camera -> world transform).
        :return:                If possible, a tuple consisting of the estimated depth image and a depth difference
                                image in which each pixel denotes the absolute difference between estimates of the
                                depth based on two different keyframes, or None otherwise.
        """
        best_depth_image: Optional[np.ndarray] = None
        second_best_depth_image: Optional[np.ndarray] = None

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
                second_best_depth_image = self.__multiview_depth_estimator.estimate_depth(
                    colour_image, second_best_keyframe_image, tracker_w_t_c, second_best_keyframe_w_t_c
                )

                # If we're debugging, show both depth images.
                if self.__debug:
                    cv2.imshow("Best Depth Image", best_depth_image / 2)
                    cv2.imshow("Second Best Depth Image", second_best_depth_image / 2)
                    cv2.waitKey(1)

        # Check whether this frame should be a new keyframe. If so, add it to the list.
        if translation_to_closest_keyframe > self.__max_translation_before_keyframe \
                or rotation_to_closest_keyframe > self.__max_rotation_before_keyframe:
            self.__keyframes.append((colour_image.copy(), tracker_w_t_c.copy()))

        # If best and second-best depth images were successfully estimated:
        if best_depth_image is not None:
            # Calculate the average of the two depth images.
            estimated_depth_image: np.ndarray = (best_depth_image + second_best_depth_image) / 2

            # Fill the border with zeros (depths around the image border are often quite noisy).
            estimated_depth_image = ImageUtil.fill_border(estimated_depth_image, self.__border_to_fill, 0.0)

            # Calculate how inconsistent the depth estimates are for each pixel.
            depth_diff_image: np.ndarray = np.abs(best_depth_image - second_best_depth_image)

            # If we're debugging, show the output images.
            if self.__debug:
                cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)
                cv2.imshow("Depth Inconsistency Image", depth_diff_image)
                cv2.waitKey(1)

            return estimated_depth_image, depth_diff_image
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
        self.__multiview_depth_estimator.set_intrinsics(intrinsics)
        return self
