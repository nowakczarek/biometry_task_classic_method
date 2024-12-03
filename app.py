import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import cv2
import os

img1 = cv2.imread('SOCOfing/real/1__M_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('SOCOfing/altered/altered_easy/1__M_Right_middle_finger_Obl.BMP', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

plot_img1 = img1.copy()
plot_img2 = img2.copy()
plot_img1 = cv2.drawKeypoints(img1, kp1, plot_img1)
plot_img2 = cv2.drawKeypoints(img2, kp2, plot_img2)

fig, axs = plt.subplots(1,2)
fig.set_size_inches(15, 5)

axs[0].imshow(plot_img1, cmap='gray')
axs[1].imshow(plot_img2, cmap= 'gray')
plt.show()


bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

matcher_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matcher_img)
plt.show()

good_matches = []
for m,n in matches:
    if m.distance < 0.9* n.distance:
        good_matches.append([m])

matcher_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matcher_img)
plt.show()

#loading the data

all_paths = []
sample_ids = []

real_path = 'SOCOfing/real/'
for finger in os.listdir(real_path):
    if 'BMP' in finger:
        all_paths.append(real_path + finger)
        sample_ids.append(finger.split('_finger')[0])

altered_paths = ['SOCOfing/altered/altered_easy/', 'SOCOfing/altered/altered_medium/', 'SOCOfing/altered/altered_hard/']

for folder in altered_paths:
    for finger in os.listdir(folder):
        if 'BMP' in finger:
            all_paths.append(folder + finger)
            sample_ids.append(finger.split('_finger')[0])


categorical_ids = preprocessing.LabelEncoder().fit_transform(sample_ids)
print(categorical_ids)

data = []
for path in all_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    data.append(img)

data = np.array(data)
data = np.expand_dims(data, -1)
print(data.shape)