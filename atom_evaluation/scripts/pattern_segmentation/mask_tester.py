import cv2
import numpy as np
import os
import re
from PIL import Image


def extract_number(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else float('inf')  # Use inf to handle unexpected filenames


images_path = "./random_images/"
# mask_path = "./random_saves/"
mask_path = "./random_gt/"
files = os.listdir(images_path)
# files.remove("annotation.json")
files = sorted(files, key=extract_number)
shadow_color = np.array([255, 0, 0], dtype=np.uint8)  # Blue color
shadow_color = np.array([0, 255, 0], dtype=np.uint8)  # Green color

for filename in files:
    # if extract_number(filename)==61:
        # img = cv2.imread(images_path + 'pattern_'+str(img_num)+'.jpg')
        # mask = cv2.imread(mask_path + 'mask_pattern_'+str(img_num)+'.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(images_path + filename)
        # mask = cv2.imread(mask_path + filename, cv2.IMREAD_GRAYSCALE)
        try:
            mask = Image.open(mask_path + filename).convert("L")
        except:
            mask = np.zeros((img.shape[0], img.shape[1]))
        # print(mask)
        mask = np.array(mask, dtype=np.uint8)
        # print(mask)
        # print(mask.shape)
        # img[mask == 255] = (0, 0, 0)
        alpha = 0.5  # Shadow intensity (0 = no change, 1 = black)
        # img[mask == 255] = (img[mask == 255] * alpha).astype(np.uint8)
        img[mask == 255] = ((1 - alpha) * img[mask == 255] + alpha * shadow_color).astype(np.uint8)
        # cv2.imwrite('./post_process/preds/Small_dataset_unet/'+filename, img)
        cv2.imwrite('./post_process/gt/'+filename, img)
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