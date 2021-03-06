from cv2 import cv2
import numpy as np


def sort_contours(cnts, img):
    sorted_cnt_x = sorted(cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])

    sorted_cnt_xy = [[]]
    x_old = 0
    for i, cnt in enumerate(sorted_cnt_x):
        x, y, w, h = cv2.boundingRect(cnt)

        if i != 0 and x > x_old + 10:
            sorted_cnt_xy.append([])

        if w > 5 * h and y > img.shape[1] // 2:
            sorted_cnt_xy[-1].append(cnt)

        x_old = x

    for cnts in sorted_cnt_xy:
        cnts.sort(key=lambda cnt: cv2.boundingRect(cnt)[1])

    sorted_cnt_xy_final = [cnt for cnts in sorted_cnt_xy for cnt in cnts]

    return sorted_cnt_xy_final

# Functon for extracting the box
def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    print("Reading image..")
    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 230, 255,
                                      cv2.THRESH_BINARY_INV)  # Thresholding the image
    # img_bin = 255 - img_bin  # Invert the image

    print("Storing binary image to Images/Image_bin.jpg..")
    cv2.imwrite("Images/Image_bin.jpg", img_bin)

    print("Applying Morphological Operations..")
    # Defining a kernel length
    kernel_length = np.array(img).shape[1] // 40

    # A verticle kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("Images/verticle_lines.jpg", verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("Images/horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    print("Binary image which only contains boxes: Images/img_final_bin.jpg")
    cv2.imwrite("Images/img_final_bin.jpg", img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours left-to-right, top-to-bottom.
    ans_cnts_sorted_xy = sort_contours(contours, img)

    idx = 1
    for cnt in ans_cnts_sorted_xy:
        x, y, w, h = cv2.boundingRect(cnt)
        new_img = img[y:y + h, x:x + w]
        cv2.imwrite(cropped_dir_path + str(idx) + '.png', new_img)
        idx += 1

    print("Output stored in Output directory!")

    # For Debugging
    # Enable this line to see all contours.
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imwrite("./Images/img_contour.jpg", img)


# Input image path and out folder
box_extraction("./images/test_1.png", "./Output/")
