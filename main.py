from cv2 import cv2
import os
from joblib import load
import numpy as np


def predict_ans(ans_img):
    ans_img = cv2.resize(ans_img, dsize=(28, 28))
    ans_img = cv2.cvtColor(ans_img, cv2.COLOR_BGR2GRAY)
    ans_img = ans_img.reshape(-1)
    result = clf.predict(np.array([ans_img]))[0]
    return result


# read img
img_original = cv2.imread('images/test.jpg')
img_original = cv2.resize(img_original, (786, 1118))

# find and draw contours
## img to gray
img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
## blur
## since edge detection is susceptible to noise in the image, we need to remove the noise in the image with a 5x5 Gaussian filter.
img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
## detect edge
img = cv2.Canny(img, threshold1=50, threshold2=200)
## find contours
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# find valid contour for 4 columns
## find contours that satisfy condition
ans_column_cnts = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if 4 < h * 1.0 / w:  # these are hard-coded values and are subject to change
        print('h: ', h)
        print('w: ', w)
        ans_column_cnts.append(contour)  # có thể thêm đoạn check nếu contour có 4 đỉnh (để chắc chắn nó là hình chữ nhật) rồi mới thêm vào

## sort contour from left to right
ans_column_cnts = sorted(ans_column_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])

## draw contour
cv2.drawContours(img_original, ans_column_cnts, -1, (0, 255, 0), 5)
assert len(ans_column_cnts) == 4, "there should be 4 answer columns"

# find answer for each question
clf = load('model.joblib')
ans_list = [[]]
question_count = 1
key_count = 1

## np.hsplit() and np.vsplit() can be used to quickly find boxes and answers
for col_cnt in ans_column_cnts:
    x, y, w, h = cv2.boundingRect(col_cnt)
    box_height = h // 6
    for box_idx in range(6):
        ## find boxes of answers. each column has 6 boxes
        box = img_original[y + box_idx * box_height:y + (box_idx + 1) * box_height, x:x + w]
        box = box[8:box.shape[0] - 7, :]  # 7 is a hard-coded value
        ## separate questions in each box
        question_height = box.shape[0] // 5
        for question_idx in range(5):
            question = box[question_idx * question_height:(question_idx + 1) * question_height, :]
            key_width = question.shape[1] // 5
            for key_idx in range(5):
                if key_idx != 0:
                    key = question[:, key_idx * key_width:(key_idx + 1) * key_width]
                    res = predict_ans(key)
                    ans_list[-1].append(res)
            ans_list.append([])

temp_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
ans_dict = {}
for idx, ans in enumerate(ans_list[:-1]):
    key = np.argmax(np.array(ans))
    if ans[key] != 0.:
        key = temp_dict[key + 1]
        ans_dict[idx + 1] = key
    else:
        ans[idx+1] = 'unknown'

for key, value in ans_dict.items():
    print(key, ': ', value)
