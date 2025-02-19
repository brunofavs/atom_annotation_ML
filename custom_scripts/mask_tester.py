import cv2
import numpy as np
import os
import re


def extract_number(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else float('inf')  # Use inf to handle unexpected filenames


# mask_path = "../sample_masks/"
# images_path = "../sample_images/"
images_path = "../segmentation_dataset_v2/train/images/"
mask_path = "../segmentation_dataset_v2/train/masks/"
files = os.listdir(images_path)
files.remove("annotation.json")
files = sorted(files, key=extract_number)

for filename in files:
    # img = cv2.imread(images_path + 'pattern_'+str(img_num)+'.jpg')
    # mask = cv2.imread(mask_path + 'mask_pattern_'+str(img_num)+'.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(images_path + filename)
    mask = cv2.imread(mask_path + filename, cv2.IMREAD_GRAYSCALE)
    # print(mask.shape)
    # img[mask == 255] = (0, 0, 0)
    alpha = 0.1  # Shadow intensity (0 = no change, 1 = black)
    img[mask == 255] = (img[mask == 255] * alpha).astype(np.uint8)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        if key == ord('e'):
            cv2.destroyAllWindows()
            exit()