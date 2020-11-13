import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from typing import List

from smg.external.mvdepthnet.depthNet_model import depthNet


class MVDepthEstimator:
    """An MVDepthNet depth estimator."""

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
    def estimate_depth(self, left_image: np.ndarray, right_image: np.ndarray,
                       left_pose: np.ndarray, right_pose: np.ndarray) -> np.ndarray:
        """
        Estimate a depth image corresponding to the left input image.

        :param left_image:  The left input image.
        :param right_image: The right input image.
        :param left_pose:   The camera pose corresponding to the left input image.
        :param right_pose:  The camera pose corresponding to the right input image.
        :return:            The estimated depth image corresponding to the left input image.
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

        # Prepare the matrices that need to be passed to the model.
        left2right: np.ndarray = np.dot(np.linalg.inv(right_pose), left_pose)
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

        # Run the model.
        outputs: List[torch.Tensor] = self.__model(
            MVDepthEstimator.__image_to_cuda_tensor(left_image),
            MVDepthEstimator.__image_to_cuda_tensor(right_image),
            KRKiUV_cuda_T, KT_cuda_T
        )

        # Get the predicted inverse depth image.
        inv_depth_image: np.ndarray = np.squeeze(outputs[0].cpu().data.numpy())

        # Invert and return it.
        return 1.0 / inv_depth_image

    # PRIVATE STATIC METHODS

    @staticmethod
    def __image_to_cuda_tensor(image: np.ndarray) -> torch.Tensor:
        """
        Convert an image to a CUDA tensor so that it can be passed to the model, normalising it in the process.

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
