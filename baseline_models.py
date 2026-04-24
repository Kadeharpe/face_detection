import os
import cv2
import numpy as np

def load_data(folder):
    X = []
    y = []

    for label in os.listdir(folder):
        path = os.path.join(folder, label)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # smaller size
            img = img.flatten()  # turn into vector

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = load_data("celeba_yolo_cls/train")
X_test, y_test = load_data("celeba_yolo_cls/val")
#loaded images ^

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nkNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0)) #precision, recall, f1
#run k-NN ^

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
from sklearn.metrics import classification_report
print("\nClassification report:")
print(classification_report(y_test, y_pred_svm, zero_division=0)) #prec recall f1
#run SVM ^