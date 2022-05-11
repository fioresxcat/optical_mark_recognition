from cv2 import cv2
from sklearn import svm
from pathlib import Path
import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score

list_ans = [2, 8, 10, 11, 13, 19, 22, 25, 32, 35, 38, 41, 47, 52, 54, 57, 63, 66, 69, 75, 78, 84, 90, 95, 103, 106, 109,
            114, 120, 126, 134, 144, 152, 153, 161, 184, 186, 202, 209, 221, 230, 261, 280, 292]

y = np.zeros((94, ))
X = []

# prepare the data
train_path = Path('./train')
for idx, img_path in enumerate(train_path.glob('*.jpg')):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(-1)
    X.append(img)
    if int(img_path.stem[4:]) in list_ans:
        y[idx] = 1

X = np.array(X)

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# SVM model
clf = svm.SVC()
hjo
# train
clf.fit(X, y)

# test
y_pred = clf.predict(X)
print(accuracy_score(y, y_pred))

# save model
dump(clf, 'model.joblib')
print('Done')

