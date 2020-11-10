import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from smg.external.mvdepthnet.depthNet_model import depthNet
from smg.external.mvdepthnet.visualize import np2Depth


class MVDepthEstimator:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, model_path: str, intrinsics: np.ndarray):
        """
        TODO

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

    def estimate_depth(self, left_image: np.ndarray, right_image: np.ndarray,
                       left_pose: np.ndarray, right_pose: np.ndarray) -> np.ndarray:
        """
        TODO

        :param left_image:  TODO
        :param right_image: TODO
        :param left_pose:   TODO
        :param right_pose:  TODO
        :return:            TODO
        """
        # TODO
        left2right: np.ndarray = np.dot(np.linalg.inv(right_pose), left_pose)

        # TODO
        # for warp the image to construct the cost volume
        pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        pixel_coordinate = np.concatenate(
            (pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

        # scale to 320x256
        original_width = left_image.shape[1]
        original_height = left_image.shape[0]
        factor_x = 320.0 / original_width
        factor_y = 256.0 / original_height

        left_image = cv2.resize(left_image, (320, 256))
        right_image = cv2.resize(right_image, (320, 256))
        camera_k = self.__intrinsics.copy()
        camera_k[0, :] *= factor_x
        camera_k[1, :] *= factor_y

        # convert to pythorch format
        torch_left_image = np.moveaxis(left_image, -1, 0)
        torch_left_image = np.expand_dims(torch_left_image, 0)
        torch_left_image = (torch_left_image - 81.0) / 35.0
        torch_right_image = np.moveaxis(right_image, -1, 0)
        torch_right_image = np.expand_dims(torch_right_image, 0)
        torch_right_image = (torch_right_image - 81.0) / 35.0

        # process
        left_image_cuda = torch.Tensor(torch_left_image).cuda()
        # left_image_cuda = Variable(left_image_cuda, volatile=True)

        right_image_cuda = torch.Tensor(torch_right_image).cuda()
        # right_image_cuda = Variable(right_image_cuda, volatile=True)

        left_in_right_T = left2right[0:3, 3]
        left_in_right_R = left2right[0:3, 0:3]
        K = camera_k
        K_inverse = np.linalg.inv(K)
        KRK_i = K.dot(left_in_right_R.dot(K_inverse))
        KRKiUV = KRK_i.dot(pixel_coordinate)
        KT = K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KT = np.expand_dims(KT, 0)
        KT = KT.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KRKiUV = np.expand_dims(KRKiUV, 0)
        KRKiUV_cuda_T = torch.Tensor(KRKiUV).cuda()
        KT_cuda_T = torch.Tensor(KT).cuda()

        predict_depths = self.__model(left_image_cuda, right_image_cuda, KRKiUV_cuda_T, KT_cuda_T)

        # visualize the results
        idepth = np.squeeze(predict_depths[0].cpu().data.numpy())
        np_depth = np2Depth(idepth, np.zeros(idepth.shape, dtype=bool))
        result_image = np.concatenate(
            (left_image, right_image, np_depth), axis=1)

        return 1.0 / idepth
