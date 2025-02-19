import cv2
import json
import numpy as np
import re

# Global variables
LOG_collection = []
img_num = None

def extract_number(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else float('inf')  # Use inf to handle unexpected filenames

def collection_error():
    print(f"The collection number {img_num} has annotation issues. Please redo the annotations for this collection")
    LOG_collection.append(img_num)


def compute_slope(point1, point2):
    """ Compute the slope of a line given two points. """
    if point2[0] - point1[0] == 0:  # Avoid division by zero (vertical line)
        return float('inf')
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

def dataset2points(img_num, dataset, img, border_error=0.01, slope_threshold=0.3):
    """
    Reads the lines' points labeled using ATOM and then corrects and add points 
    to form a fully defined polygon to form a mask of the checkerboard.
    
    Args:
        img_num: Image index in the dataset.
        dataset: Dictionary resultant from the annotations' json
        img: The image array (to get dimensions).
        border_error: Tolerance to snap points to image borders.
        slope_treshold: The tolerance for considering two lines as parallel ones.

    Returns:
        corrected_points: List of (x, y) points forming the board mask.
    """
    W, H = img.shape[1], img.shape[0]  # Image width and height
    x_tol = border_error * W
    y_tol = border_error * H

    # Extract line data
    line_keys = ["top", "right", "bottom", "left"]  # Board sides (since the camera may be in a diferrent orientation, we cannot rely on these sides)
    line_list = []
    slopes = {}
    present_sides = []
    for i, key in enumerate(line_keys):
        if key in dataset[str(img_num)]:
            xs = dataset[str(img_num)][key]["xs"]
            ys = dataset[str(img_num)][key]["ys"]
            if len(xs) == 1 or len(ys) == 1:
                print("ERROR: Line with only one point defined")
                collection_error()
                return None  
            if len(xs) == 2 and len(ys) == 2:
                line = [[xs[0], ys[0]], [xs[1], ys[1]]]
                line_list.append([[xs[0], ys[0]], [xs[1], ys[1]]])
                slopes[chr(65 + i)] = compute_slope(line[0], line[1])
                present_sides.append(chr(65 + i))

    # Extract all points
    if len(present_sides)==0:
        return -1
    points_list = [pt for line in line_list for pt in line]

    if len(present_sides) == 2:
        """
        If only two sides are present:
        - Use slopes to check if the sides are parallel;
        - If lines are parallel, check if there is a corner of the image in the
        middle. If there is a corner, add the corner to the point list.
        # TODO This doesn't take into account if there are two corners between
        the lines, which may happen
        - If lines are NOT parallel, compute the missing corner.
        """
        side_1, side_2 = present_sides
        slope_1 = slopes[side_1]
        slope_2 = slopes[side_2]

        # Extract x and y values of known points
        known_x = [x for x, y in points_list]
        known_y = [y for x, y in points_list]

        min_x, max_x = min(known_x), max(known_x)
        min_y, max_y = min(known_y), max(known_y)

        # If slopes are similar, lines are parallel
        if abs(slope_1 - slope_2) < slope_threshold:

            # Check if the parallel lines cross different edges
            crosses_left = min_x < x_tol
            crosses_right = max_x > (W - x_tol)
            crosses_top = min_y < y_tol
            crosses_bottom = max_y > (H - y_tol)

            # Step 3: Find the missing corner in diagonal cases
            missing_x, missing_y = None, None

            if crosses_left and crosses_bottom:
                missing_x, missing_y = 0, H
            elif crosses_bottom and crosses_right:
                missing_x, missing_y = W, H
            elif crosses_right and crosses_top:
                missing_x, missing_y = W, 0
            elif crosses_top and crosses_left:
                missing_x, missing_y = 0, 0

            if missing_x is not None and missing_y is not None:
                points_list.append([missing_x, missing_y])

        else: # If slopes are different, lines are most likely perpendicular and there will be a corner missing
            missing_x, missing_y = None, None

            # Determine missing corner
            if min_x < x_tol:
                missing_x = 0
            elif max_x > (W - x_tol):
                missing_x = W

            if min_y < y_tol:
                missing_y = 0
            elif max_y > (H - y_tol):
                missing_y = H

            if missing_x is not None and missing_y is not None:
                points_list.append([missing_x, missing_y])

    elif len(present_sides) == 3:
        """
        If three sides are present:
        - Identify the two parallel lines.
        - If they meet different edges, add the missing image corner.
        """
        # ##### Find the two parallel lines (closest slopes) #####
        # sorted_sides = sorted(slopes.items(), key=lambda x: x[1])  # Sort by slope
        # best_pair = None
        # min_slope_diff = float('inf')

        # for i in range(len(sorted_sides) - 1):
        #     slope_diff = abs(sorted_sides[i][1] - sorted_sides[i + 1][1])
        #     if slope_diff < min_slope_diff:
        #         min_slope_diff = slope_diff
        #         best_pair = (sorted_sides[i][0], sorted_sides[i + 1][0])  # Parallel sides

        # parallel_sides = set(best_pair)
        # missing_side = list(set(present_sides) - parallel_sides)[0]  # The non-parallel side



        # Experimenting without checking which ones are parallel...


        ##### Check which edges the board intersects #####
        known_x = [x for x, y in points_list]
        known_y = [y for x, y in points_list]

        min_x, max_x = min(known_x), max(known_x)
        min_y, max_y = min(known_y), max(known_y)

        missing_x, missing_y = None, None

        # Check if the parallel lines cross different edges
        crosses_left = min_x < x_tol
        crosses_right = max_x > (W - x_tol)
        crosses_top = min_y < y_tol
        crosses_bottom = max_y > (H - y_tol)

        ##### Find the missing corner in diagonal cases #####
        if crosses_left and crosses_bottom:
            missing_x, missing_y = 0, H
        elif crosses_bottom and crosses_right:
            missing_x, missing_y = W, H
        elif crosses_right and crosses_top:
            missing_x, missing_y = W, 0
        elif crosses_top and crosses_left:
            missing_x, missing_y = 0, 0

        if missing_x is not None and missing_y is not None:
            points_list.append([missing_x, missing_y])

    # Apply border correction
    corrected_points = []
    for x, y in points_list:
        x = 0 if x <= x_tol else (W if x >= (W - x_tol) else x)
        y = 0 if y <= y_tol else (H if y >= (H - y_tol) else y)
        corrected_points.append([x, y])

    return corrected_points

# img_num = 11
# img_num = 76
# img_num = 0
# img_num = 12

# path = "../sample_images/"
path = "../segmentation_dataset_v2/train/images/"
masks_path = "../segmentation_dataset_v2/train/masks/"
json_file = path + "annotation.json"
f = open(json_file, 'r')
dataset = json.load(f)
f.close()  # Close the file to prevent memory leaks


import os
files = os.listdir(path)
files.remove("annotation.json")
files = sorted(files, key=extract_number)

# for img_num in range(1, 77):
for filename in files:
    match = re.search(r'_(\d+)\.jpg$', filename)
    img_num = int(extract_number(filename))
    img = cv2.imread(path + filename)
    points_list = dataset2points(img_num, dataset, img, border_error=0.005, slope_threshold=0.5)
    if points_list is not None:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if points_list != -1:
            # Define the polygon points
            points = np.array(points_list, np.int32)
            # Fill the polygon
            cv2.fillPoly(mask, [cv2.convexHull(points)], 255)
        cv2.imwrite(masks_path + filename, mask)


# img = cv2.imread(path + 'pattern_'+str(img_num)+'.jpg')
# points_list = dataset2points(img_num, dataset, img, border_error=0.005, slope_threshold=0.5)
# if points_list is not None:
#     mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#     # Define the polygon points
#     points = np.array(points_list, np.int32)
#     # Fill the polygon
#     cv2.fillPoly(mask, [cv2.convexHull(points)], 255)
#     # img[mask == 255] = (0, 0, 0)
#     cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#     cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
#     cv2.imshow("Image", img)
#     cv2.imshow('Mask', mask)
#     while True:
#         key = cv2.waitKey(0) & 0xFF
#         if key == ord('q'):
#             cv2.destroyAllWindows()
#             break




if len(LOG_collection)>0:
    # open file
    with open('log.txt', 'w+') as f:
        # write elements of list
        for items in LOG_collection:
            f.write('%s\n' %items)
        print("The collection numbers with errors are saved at log.txt")
    # close the file
    f.close()  






