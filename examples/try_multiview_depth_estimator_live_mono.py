import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Dict, Optional

from smg.imagesources import RGBFromRGBDImageSource, RGBImageSource
from smg.mvdepthnet import MultiviewDepthEstimator
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera, OpenNIRGBDImageSource
from smg.pyorbslam2 import MonocularTracker
from smg.rotory import DroneFactory, DroneRGBImageSource
from smg.utility import GeometryUtil


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str,
        help="an optional directory into which to save the keyframes"
    )
    parser.add_argument(
        "--source_type", "-t", type=str, required=True, choices=("ardrone2", "kinect", "tello"),
        help="the source type"
    )
    args: dict = vars(parser.parse_args())

    image_source: Optional[RGBImageSource] = None
    try:
        # Construct the RGB image source.
        # FIXME: This is duplicate code - factor it out.
        source_type: str = args["source_type"]
        if source_type == "kinect":
            image_source = RGBFromRGBDImageSource(OpenNIRGBDImageSource(OpenNICamera(mirror_images=True)))
        else:
            kwargs: Dict[str, dict] = {
                "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
                "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
            }
            image_source = DroneRGBImageSource(DroneFactory.make_drone(source_type, **kwargs[source_type]))

        # Construct the tracker.
        with MonocularTracker(
            settings_file=f"settings-{source_type}.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            # Construct the depth estimator.
            depth_estimator: MultiviewDepthEstimator = MultiviewDepthEstimator(
                "C:/Users/Stuart Golodetz/Downloads/MVDepthNet/opensource_model.pth.tar"
            ).set_intrinsics(GeometryUtil.intrinsics_to_matrix(image_source.get_intrinsics()))

            reference_image: Optional[np.ndarray] = None
            reference_pose: Optional[np.ndarray] = None
            estimated_depth_image: Optional[np.ndarray] = None

            while True:
                # Get the colour image from the camera, and show it.
                colour_image: np.ndarray = image_source.get_image()
                cv2.imshow("Colour Image", colour_image)
                c: int = cv2.waitKey(1)

                # If the tracker's not yet ready, or the pose can't be estimated for this frame, continue.
                if not tracker.is_ready():
                    continue

                pose: Optional[np.ndarray] = tracker.estimate_pose(colour_image)
                if pose is None:
                    continue

                # If the user presses the 'r' key, set this frame as the reference.
                if c == ord('r'):
                    reference_image = colour_image.copy()
                    reference_pose = pose.copy()
                    continue

                # If the user presses the 'v' key, exit the loop so that the estimated depth image can be visualised.
                if c == ord('v'):
                    break

                # Provided the reference frame has been set:
                if reference_image is not None:
                    # Estimate a depth image for the current frame, and show it.
                    estimated_depth_image = depth_estimator.estimate_depth(
                        colour_image, reference_image, np.linalg.inv(pose), np.linalg.inv(reference_pose)
                    )
                    cv2.imshow("Estimated Depth Image", estimated_depth_image / 2)
                    cv2.waitKey(1)

            # Visualise the 3D point cloud corresponding to the most recently estimated depth image (if any).
            if estimated_depth_image is not None:
                VisualisationUtil.visualise_rgbd_image(
                    colour_image, estimated_depth_image, image_source.get_intrinsics()
                )
    finally:
        # Terminate the image source once we've finished.
        if image_source is not None:
            image_source.terminate()


if __name__ == "__main__":
    main()
