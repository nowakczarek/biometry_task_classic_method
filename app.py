import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import cv2
import os


# load data in np form labels and images
# def load_data_nn():
#     all_paths = []
#     sample_ids = []

#     real_path = 'SOCOfing/real/'
    # for finger in os.listdir(real_path):
    #     if 'BMP' in finger:
    #         all_paths.append(real_path + finger)
    #         sample_ids.append(finger.split('_finger')[0])

#     altered_paths = ['SOCOfing/altered/altered_easy/', 'SOCOfing/altered/altered_medium/', 'SOCOfing/altered/altered_hard/']

    # for folder in altered_paths:
    #     for finger in os.listdir(folder):
    #         if 'BMP' in finger:
    #             all_paths.append(folder + finger)
    #             sample_ids.append(finger.split('_finger')[0])
    
    # images = np.array(all_paths)
    # labels = np.array(sample_ids)

#     return images, labels


#load data for classic method
def load_data(images_path):
    images = []
    for file in os.listdir(images_path):
        img = cv2.imread(images_path + file, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    return images

def preprocess_image(image):
    #adaptive thresholding of image
    maximum_value = np.max(image)
    threshold_type = cv2.THRESH_BINARY
    threshold_value = 125
    blocksize = 11
    C = 2

    binary_image = cv2.adaptiveThreshold(image, maximum_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, blocksize, C)

    #setting up gabor filtering
    sigma = 1
    kernel_size = (5,5)
    theta = 0
    lambd = 5
    gamma = 0.5
    psi = 1
    kernel = cv2.getGaborKernel(kernel_size, sigma, theta, lambd, gamma, psi)

    #implementing gabor filtering
    gabor_img = cv2.filter2D(binary_image, cv2.CV_32F, kernel)

    gabor_img_uint8 = np.uint8(gabor_img * 255 / np.max(gabor_img))

    return gabor_img_uint8

def detect_and_match_fingers(image1, image2):
    sift = cv2.SIFT_create()

    keypoint1, descriptor1 = sift.detectAndCompute(image1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(image2, None)

    # plot_img1 = img1.copy()
    # plot_img2 = img2.copy()
    # plot_img1 = cv2.drawKeypoints(image1, keypoint1, plot_img1)
    # plot_img2 = cv2.drawKeypoints(image2, keypoint2, plot_img2)
  
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    # matcher_img = cv2.drawMatchesKnn(image1, keypoint1, image2, keypoint2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # plt.imshow(matcher_img)
    # plt.show()

    return matches

def calculate_matches(matches):
    good_matches = []
    for m,n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append([m])
    return good_matches

# all_images = load_data('SOCOFING/real/')
# altered_easy = load_data('SOCOFING/altered/altered_easy/')

# all = altered_easy + all_images

# processed_images = [preprocess_image(img) for img in all]

# score = []

# for i in range(len(processed_images) - 1):
#     for j in range(i + 1, len(processed_images)):
#         matches = detect_and_match_fingers(processed_images[i], processed_images[j])
#         good_matches = calculate_matches(matches)
#         accuracy = len(good_matches)/len(matches)
#         score.append(accuracy)
#         print(f"Pair {i} - {j} : Accuracy: {accuracy:.2f}%")

# overall_score = np.mean(score)
# print(f'Overall Score: {overall_score: .2f} %')

img1 = cv2.imread('SOCOfing/altered/altered_hard/2__F_Right_thumb_finger_Obl.BMP', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('SOCOfing/altered/altered_hard/2__F_Right_thumb_finger_Zcut.BMP', cv2.IMREAD_GRAYSCALE)

img_test = preprocess_image(img1)
img_test2 = preprocess_image(img2)

matches = detect_and_match_fingers(img_test, img_test2)
good_matches = calculate_matches(matches)

print(len(good_matches))
print(len(matches))
