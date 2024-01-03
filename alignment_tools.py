""" this module provides image processing and analysis tools
    for camera auto-alignments

    Methods:
        filename_parsing():     parsing the file name to extract robot pose
        fiducial_detection2():  detect feature points from an image
        query_dark_location():  find dark feature points
        query_light_location(): find white feature points
        fiducial_points():      find feature points
        angle_estimation():     estimage angles
        center_estimation():    estimage center location
        distance_estimation():  estimage distnace ebtween feature points
        process_image():        use related methods to estimate
        analysis_result():      find the maximum relative error and its index
"""


import os
import logging
import Type

import numpy as np
import pandas as pd
import cv2

RADIUS_FACTOR = 4               # determine radius threshold
AREA_FACTOR = 80                # determine area threshold
MIN_GRAYSCALE_THRESHOLD = 20    # min threshold
MAX_GRAYSCALE_THRESHOLD = 250   # max threshold
MIN_CIRCULARITY1 = 0.8          # circularity threshold for detector 1
MIN_CONVEXITY1 = 0.9            # convexity threshold for detector 1
MIN_INERTIAL_RATIO = 0.02       # inertial threshold
MAXIMUM_GRAYSCALE = 255         # max grayscale for the 8-bits image
MIN_CIRCULARITY2 = 0.7          # circularity threshold for detector 2
MIN_CONVEXITY2 = 0.85           # convexity threshold for detector 2

# setup the logger
logger = logging.getLogger("auto-alignment logger")


def filename_parsing(file_name: str) -> pd.DataFrame:
    """ sparse the filename to extract the robot pos

    Args:
        file_name: the file name with the robot pose information

    Returns:
        robot pose extracted from the the file name
    """

    pose_from_file = []
    file_str = file_name.split("_")
    pose_from_file.append(str(file_str[-2]))
    pose_from_file.append(str(file_str[-1].split(".")[0]))
    pose_from_file.append(float(file_str[1].replace("p", ".")))
    pose_from_file.append(float(file_str[3].replace("p", ".")))
    pose_from_file.append(float(file_str[5].replace("p", ".")))
    pose_from_file.append(float(file_str[7].replace("p", ".")))
    pose_from_file.append(float(file_str[9].replace("p", ".")))
    pose_from_file.append(float(file_str[11].replace("p", ".")))
    pose_from_file.append(float(file_str[13].replace("p", ".")))

    return pose_from_file


def fiducial_detection2(detector: Type(cv2.detector), img: np.ndarray
                        ) -> np.ndarray:
    """  detect feature points from an image

    Args:
        detector:   blob detector in open CV
        img:        the input image

    Returns:
        feature points
    """

    keypoints = detector.detect(img)
    fiducials = np.zeros((len(keypoints), 2))
    for idx, keypoint in enumerate(keypoints):
        fiducials[idx, 0] = keypoint.pt[0]
        fiducials[idx, 1] = keypoint.pt[1]
    return fiducials


def query_dark_location(data: np.ndarray, reference_pt: np.ndarray
                        ) -> np.ndarray:
    """ find 4 dark feature points

    Args:
        data:           extracted points
        reference_pt:   threshold to identify points for centering analysis

    Returns:
        dark feature points
    """

    # search left dark circle
    x = data[:, 0] - reference_pt[0] / 2
    y = data[:, 1]
    l2distance = np.sqrt(x**2 + y**2)
    idx = np.argmin(l2distance)
    leftdark = data[idx, :]

    # search top dark circle
    x = data[:, 0]
    y = data[:, 1] - reference_pt[1] / 2
    l2distance = np.sqrt(x**2 + y**2)
    idx = np.argmin(l2distance)
    topdark = data[idx, :]

    # search right dark circle
    x = data[:, 0] - reference_pt[0] / 2
    y = data[:, 1] - reference_pt[1]
    l2distance = np.sqrt(x**2 + y**2)
    idx = np.argmin(l2distance)
    rightdark = data[idx, :]

    # search bottom dark circle
    x = data[:, 0] - reference_pt[0]
    y = data[:, 1] - reference_pt[1] / 2
    l2distance = np.sqrt(x**2 + y**2)
    idx = np.argmin(l2distance)
    bottomdark = data[idx, :]

    logging.info("query_dark_location found.")
    return np.vstack((leftdark, topdark, rightdark, bottomdark))


def query_light_location(data: np.ndarray, threshold: int) -> np.ndarray:
    """ find bright feature points

    Args:
        data:       extracted points
        threshold:  threshold to identify points for centering analysis

    Returns:
        bright feature points
    """

    reference_pt = np.mean(data, 0)
    x = data[:, 0] - reference_pt[0]
    y = data[:, 1] - reference_pt[1]
    l2distance = np.sqrt(x**2 + y**2)
    mask = l2distance < threshold
    center_pts = data[mask, :]
    if center_pts[0, 0] >= center_pts[1, 0]:
        pt1 = center_pts[0, :]
        pt2 = center_pts[1, :]
    else:
        pt2 = center_pts[0, :]
        pt1 = center_pts[1, :]
    center = np.vstack((pt1, pt2))

    four_corners = np.vstack((x[~mask], y[~mask])).T
    topleft = np.zeros((1, 2))
    bottomleft = np.zeros((1, 2))
    topright = np.zeros((1, 2))
    bottomright = np.zeros((1, 2))
    for pt in four_corners:
        if pt[0] <= 0 and pt[1] <= 0:
            topleft = pt
        if pt[0] > 0 >= pt[1]:
            bottomleft = pt
        if pt[0] > 0 and pt[1] > 0:
            bottomright = pt
        if pt[0] <= 0 < pt[1]:
            topright = pt

    four_corners = np.vstack((topleft, topright, bottomright, bottomleft))
    four_corners[:, 0] += reference_pt[0]
    four_corners[:, 1] += reference_pt[1]

    logging.info("query_dark_location found.")
    return np.vstack((center, four_corners))


def fiducial_points(img: np.array) -> np.ndarray:
    """ find all feature points from the image

    Args:
        img: the image file to process

    Returns:
        feature points
    """

    height, width = img.shape[:2]
    radius = width if width < height else height
    radius = radius / RADIUS_FACTOR
    area = radius * radius/AREA_FACTOR

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = MIN_GRAYSCALE_THRESHOLD
    params.maxThreshold = MAX_GRAYSCALE_THRESHOLD

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = MIN_CIRCULARITY1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = MIN_CONVEXITY1

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = MIN_INERTIAL_RATIO

    # Filter by Area
    params.filterByArea = True
    params.maxArea = area

    # Create a detector with the parameters
    detector1 = cv2.SimpleBlobDetector_create(params)
    light_fiducials = fiducial_detection2(detector1, MAXIMUM_GRAYSCALE - img)
    light_locations = np.zeros((6, 2))
    if len(light_fiducials) > 6:
        logging.error("light fiducials detection error")
        light_locations = query_light_location(light_fiducials[:6, :],
                                              threshold=radius)
    else:
        light_locations = query_light_location(light_fiducials,
                                               threshold=radius)

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = MIN_CIRCULARITY2

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = MIN_CONVEXITY2

    # Create a detector with the parameters
    detector2 = cv2.SimpleBlobDetector_create(params)
    dark_fiducials = fiducial_detection2(detector2, img)
    dark_locations = np.zeros((4, 2))
    if len(dark_fiducials) > 4:
        dark_locations = query_dark_location(dark_fiducials[1:],
                                             [height, width])
    else:
        dark_locations = query_dark_location(dark_fiducials, [height, width])

    logging.info("fiducial_points found.")
    return np.vstack((light_locations, dark_locations))


def angle_estimation(data: np.ndarray) -> np.ndarray:
    """ Calculate angular features

    Args:
        data:  feature points for angular error analysis

    Return:
        angular error
    """

    logging.debug(data)
    horiz_angle = np.rad2deg(
        np.arctan((data[2, 0] - data[0, 0]) / (data[2, 1] - data[0, 1]))
    )
    verti_angle = -np.rad2deg(
        np.arctan((data[3, 1] - data[1, 1]) / (data[3, 0] - data[1, 0]))
    )
    return np.hstack((horiz_angle, verti_angle))


def center_estimation(data: np.ndarray) -> np.ndarray:
    """ Calculate center features

    Args:
        data:  feature points for centering error analysis

    Return:
        center error
    """

    logging.debug(data)
    return np.mean(data, 0)


def distance_estimation(data: np.ndarray) -> np.ndarray:
    """ Calculate distance features

    Args:
        data:  feature points for distance error analysis

    Return:
        distance error
    """

    logging.debug(data)
    top_distance = np.sqrt(
        (data[0, 0] - data[1, 0]) ** 2 + (data[0, 1] - data[1, 1]) ** 2
    )
    right_distance = np.sqrt(
        (data[1, 0] - data[2, 0]) ** 2 + (data[1, 1] - data[2, 1]) ** 2
    )
    bottom_distance = np.sqrt(
        (data[2, 0] - data[3, 0]) ** 2 + (data[2, 1] - data[3, 1]) ** 2
    )
    left_distance = np.sqrt(
        (data[3, 0] - data[0, 0]) ** 2 + (data[3, 1] - data[0, 1]) ** 2
    )
    return np.hstack((top_distance, bottom_distance, left_distance,
                      right_distance))


def process_image(folder: str, file_name: str) -> pd.DataFrame:
    """ Process the image for alignment analysis

    Args:
        folder:     folder of the image file to process
        file_name:  name of the image file to process

    Return:
        processed_results: data with robot pose and alignment features
    """

    fileinfo = filename_parsing(file_name)
    img8 = cv2.imread(os.path.join(folder, file_name), -1)
    pts = fiducial_points(img8)
    center_info = center_estimation(pts[:2, :])
    angle_info = angle_estimation(pts[6:, :])
    distance_info = distance_estimation(pts[2:6, :])

    processed_results = fileinfo
    processed_results.extend(center_info)
    processed_results.extend(distance_info)
    processed_results.extend(angle_info)

    logging.info("Image is processed.")
    return processed_results


def analysis_result(results: Type(list), threshold: Type(list),
                    desired_x: float, desired_y: float
                    ) -> (int, float):
    """ use threshold tolerance and processed results to find the maximum error

    Args:
        results:    Actual features after image processing
        threshold:  tolerance of the alignment parameters
        desired_x:  desired X (column) position of the target center
        desired_y:  desired Y (row) position of the target center

    Return:
        max_error:  relative maximum error - actual error vs tolerance
        para_index: the index corresponding to the maximum error.
                    Return -1 if aligned
    """

    para_index = -1  # if all criteria for good alignment are met

    result = results[-1]

    # calculated errors
    error_weight = [0, 0, 0, 0, 0]
    error_weight[0] = abs(result[9] - desiredx)/threshold[0]   # robot_x
    error_weight[1] = abs(result[10] - desiredy)/threshold[1]   # robot_z
    error_weight[2] = abs(result[13] - result[14])/threshold[2]   # robot_Roll
    error_weight[3] = abs(result[11] - result[12])/threshold[3]   # robot_Yaw
    error_weight[4] = abs(result[15] - 0.0)/threshold[4]   # for robot_Pitch

    # find the most significant error and set which robot DOF to adjustment
    max_error = max(error_weight)
    if max_error <= 1:   # meet all alignment requirements
        para_index = -1         # set ending signal with index less than 0
    else:
        para_index = error_weight.index(max_error)

    logging.info("Analysis results: para_index= %d, max_error = %f",
                 para_index, max_error)

    return para_index, max_error


if __name__ == "__main__":
    """ setup the logger and enter point for testing """

    logger = logging.getLogger("alignment_tools logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(".\\logger.log2")
    formatter = logging.Formatter("%(asctime)s %(name)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    filepath = "/home/rift/Downloads/Data/"
    filename = ("Rail_0p000_X_78p0000_Y_-973p0000_Z_508p0000_R_-3p780_P" +
                "_-4p615_Y_-7p250_1201-1-00027-29241.432_1694043172.712688.png")

    ret = process_image(filepath, filename)
    print(ret)
