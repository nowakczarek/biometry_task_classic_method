import numpy as np 
import matplotlib.pyplot as plt
import cv2

# image = cv2.imread('michal.jpg')
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

# plt.imshow(image)
# plt.show()

# plt.imshow(image_rgb[:, :, 2], cmap = 'gray')
# plt.show()


# v2

# image = cv2.imread('michal.jpg', cv2.IMREAD_GRAYSCALE)

# maximum_value = np.max(image)
# threshold_type = cv2.THRESH_BINARY_INV
# threshold_value = 125

# ret, binary_image = cv2.threshold(image, threshold_value, maximum_value, threshold_type)
# plt.imshow(binary_image, cmap='gray')
# plt.show()

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
