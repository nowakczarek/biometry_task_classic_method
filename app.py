import numpy as np 
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('real/1__M_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('altered/altered_medium/1__M_Left_index_finger_CR.BMP', cv2.IMREAD_GRAYSCALE)

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(15, 5)

axs[0].imshow(img1, cmap='gray')
axs[1].imshow(img2, cmap='gray')

plt.show()

sift = cv2.SIFT_create()
kp = sift.detect(img1)

plot_img = img1.copy()
plot_img = cv2.drawKeypoints(img1, kp, plot_img)

plt.imshow(plot_img)
plt.show()
