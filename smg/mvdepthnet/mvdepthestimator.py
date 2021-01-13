from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from typing import List, Optional

from smg.external.mvdepthnet.depthNet_model import depthNet


class MVDepthEstimator:
    """An MVDepthNet depth estimator."""

    # NESTED TYPES

    class CostVolumeAggregator:
        """Used to aggregate multiple cost volumes so that they can be averaged."""

        # CONSTRUCTOR

        def __init__(self):
            """Construct a cost volume aggregator."""
            self.__cost_volume: Optional[torch.Tensor] = None
            self.__count: int = 0

        # PUBLIC METHODS

        def add_cost_volume(self, cost_volume: torch.Tensor) -> MVDepthEstimator.CostVolumeAggregator:
            """
            Add a cost volume to the aggregator.

            :param cost_volume: The cost volume.
            :return:            The aggregator (to allow chaining).
            """
            if self.__cost_volume is None:
                self.__cost_volume = cost_volume
            else:
                self.__cost_volume = torch.add(self.__cost_volume, cost_volume)

            self.__count += 1

            return self

        def get_average_cost_volume(self) -> torch.Tensor:
            """
            Get the average of all the cost volumes added to the aggregator.

            :return:    The average of all the cost volumes added to the aggregator.
            """
            return self.__cost_volume / self.__count

    # CONSTRUCTOR

    def __init__(self, model_path: str, intrinsics: np.ndarray):
        """
        Construct an MVDepthNet depth estimator.

        :param model_path:  The path to the MVDepthNet model.
        :param intrinsics:  The 3x3 camera intrinsics matrix.
        """
        self.__intrinsics: np.ndarray = intrinsics

        # Load the MVDepthNet model.
        self.__model: depthNet = depthNet()
        data: dict = torch.load(model_path)
        self.__model.load_state_dict(data["state_dict"])
        self.__model = self.__model.cuda()
        cudnn.benchmark = True
        self.__model.eval()

    # PUBLIC METHODS

    # noinspection PyPep8Naming
    def estimate_depth(self, reference_image: np.ndarray, measurement_image: np.ndarray,
                       world_from_reference: np.ndarray, world_from_measurement: np.ndarray) -> np.ndarray:
        """
        Estimate a depth image corresponding to a reference image from two images and their poses.

        :param reference_image:         The reference image.
        :param measurement_image:       The measurement image.
        :param world_from_reference:    The camera pose corresponding to the reference image.
        :param world_from_measurement:  The camera pose corresponding to the measurement image.
        :return:                        The estimated depth image corresponding to the reference image.
        """
        cost_volume: torch.Tensor = self.make_cost_volume(
            reference_image, measurement_image, world_from_reference, world_from_measurement
        )
        return self.estimate_depth_from_cost_volume(reference_image, cost_volume)

    def estimate_depth_from_cost_volume(self, reference_image: np.ndarray, cost_volume: torch.Tensor) -> np.ndarray:
        """
        Estimate a depth image corresponding to a reference image from the image itself and a cost volume.

        :param reference_image: The reference image.
        :param cost_volume:     The cost volume.
        :return:                The estimated depth image corresponding to the reference image.
        """
        # Record the original reference image size.
        height, width = reference_image.shape[:2]

        # Resize the reference image to 320x256.
        reference_image = cv2.resize(reference_image, (320, 256))

        # Run the model.
        outputs: List[torch.Tensor] = self.__model.predictDepths(
            MVDepthEstimator.__image_to_cuda_tensor(reference_image), cost_volume
        )

        # Get the predicted inverse depth image.
        inv_depth_image: np.ndarray = np.squeeze(outputs[0].cpu().data.numpy())

        # Invert it, resize it to the original image size, and return it.
        depth_image: np.ndarray = 1.0 / inv_depth_image
        depth_image = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_NEAREST)
        return depth_image

    # noinspection PyPep8Naming
    def make_cost_volume(self, left_image: np.ndarray, right_image: np.ndarray,
                         world_from_left: np.ndarray, world_from_right: np.ndarray) -> torch.Tensor:
        """
        Make a cost volume that can later be used by MVDepthNet to estimate a depth image for the reference image.

        .. note::
            The parameter naming scheme here (i.e. left/right) is for consistency with the MVDepthNet code,
            but the comments are intended to make things a bit clearer.

        :param left_image:          The reference image.
        :param right_image:         The measurement image.
        :param world_from_left:     The camera pose corresponding to the reference image.
        :param world_from_right:    The camera pose corresponding to the measurement image.
        :return:                    The estimated depth image corresponding to the reference image.
        """
        # Note: Borrowed (with mild adaptations) from example2.py in the MVDepthNet code.

        # Scale the camera intrinsics prior to resizing the input images to 320x256.
        K: np.ndarray = self.__intrinsics.copy()
        K[0, :] *= 320.0 / left_image.shape[1]
        K[1, :] *= 256.0 / left_image.shape[0]

        # Resize the input images to 320x256.
        left_image = cv2.resize(left_image, (320, 256))
        right_image = cv2.resize(right_image, (320, 256))

        # For warping the image to construct the cost volume.
        pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        pixel_coordinate = np.concatenate((pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

        # Prepare the matrices that are needed for calculating the cost volume.
        left2right: np.ndarray = np.dot(np.linalg.inv(world_from_right), world_from_left)
        left_in_right_T = left2right[0:3, 3]
        left_in_right_R = left2right[0:3, 0:3]
        K_inv = np.linalg.inv(K)
        KRK_i = K.dot(left_in_right_R.dot(K_inv))
        KRKiUV = KRK_i.dot(pixel_coordinate)
        KT = K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KT = np.expand_dims(KT, 0)
        KT = KT.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KRKiUV = np.expand_dims(KRKiUV, 0)
        KRKiUV_cuda_T = torch.Tensor(KRKiUV).cuda()
        KT_cuda_T = torch.Tensor(KT).cuda()

        # Calculate the cost volume.
        return self.__model.getVolume(
            MVDepthEstimator.__image_to_cuda_tensor(left_image),
            MVDepthEstimator.__image_to_cuda_tensor(right_image),
            KRKiUV_cuda_T, KT_cuda_T
        )

    # PRIVATE STATIC METHODS

    @staticmethod
    def __image_to_cuda_tensor(image: np.ndarray) -> torch.Tensor:
        """
        Convert an image to a CUDA tensor so that it can be used by MVDepthNet, normalising it in the process.

        :param image:   The image to convert.
        :return:        The CUDA tensor.
        """
        # Reshape the 256x320x3 image to 3x256x320.
        torch_image: np.ndarray = np.moveaxis(image, -1, 0)

        # Reshape the 3x256x320 image to 1x3x256x320.
        torch_image = np.expand_dims(torch_image, 0)

        # Suitably normalise the image for MVDepthNet.
        torch_image = (torch_image - 81.0) / 35.0

        # Convert the image to a CUDA tensor and return it.
        return torch.Tensor(torch_image).cuda()
