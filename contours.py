import numpy as np
import cv2
from skimage.measure import find_contours
from pathlib import Path



if __name__ == "__main__":
    file_root = Path("/home/jaypark/Desktop/dudunet_dataset/mask")
    image_list = [name for name in file_root.glob("*.jpeg")]
    image_list.sort()

    for image_name in image_list:
        mask_image = cv2.imread(str(image_name))
        mask_image_bin = np.mean(mask_image, axis=2)
        mask_image_bin = (mask_image_bin == 255).astype(np.float)

        contours = find_contours(mask_image_bin, 0.5)
        
        contour_img = np.zeros_like(mask_image)

        for contour in contours:
           for y, x in contour:
               x, y = map(int, (x, y))
               contour_img = cv2.circle(contour_img, (x, y), 3, 255)

        cv2.imshow("contour", contour_img)
        cv2.imshow("origin", mask_image)
        cv2.waitKey(0)
