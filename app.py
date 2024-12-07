import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os

#load data for classic method
def load_data(images_path):
    images = []
    images_filenames = []
    for file in os.listdir(images_path):
        if 'BMP' in file:
            img = cv2.imread(images_path + file, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            images_filenames.append(file)
            
    return images_filenames, images

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
    kernel_size = (1,1)
    lambd = 6
    gamma = 1
    psi = 1

    #setting up different orientations to pick the best resulting one
    orientations = np.arange(0, np.pi, np.pi/4)

    kernel_bank = [cv2.getGaborKernel(kernel_size, sigma, o, lambd, gamma, psi) for o in orientations]

    #implementing gabor filtering
    filtered = [cv2.filter2D(binary_image, cv2.CV_32F, kernel) for kernel in kernel_bank]

    merged = np.max(filtered, axis=0)
    gabor_img_uint8 = np.uint8(merged * 255 / np.max(merged))

    return gabor_img_uint8

def detect_and_match_fingers(image1, image2):
    sift = cv2.SIFT_create()

    keypoint1, descriptor1 = sift.detectAndCompute(image1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    # for matching of two fingerprints for testing
     
    # plot_img1 = image1.copy()
    # plot_img2 = image2.copy()
    # plot_img1 = cv2.drawKeypoints(image1, keypoint1, plot_img1)
    # plot_img2 = cv2.drawKeypoints(image2, keypoint2, plot_img2)

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


def calculating_matches_between_images(path1, images1, path2, images2):
    scores = []
    for real_image_path, real_image in enumerate(images1):
        real_name = path1[real_image_path].split('.')[0]
        print(f'Matching for {real_name}')

        for altered_image_path, altered_image in enumerate(images2):
            if real_name in path2[altered_image_path]:
                matches = detect_and_match_fingers(real_image, altered_image)
                good_matches = calculate_matches(matches)
                accuracy = len(good_matches) / len(matches)
                scores.append(accuracy)
                print(f"Accuracy for set {real_image_path} - {altered_image_path} = {accuracy: .2f}")

    return scores


real_images_path, real_images = load_data('SOCOfing/real/')
altered_easy_images_path, altered_easy  = load_data('SOCOfing/altered/altered_easy/')
altered_medium_images_path, altered_medium = load_data('SOCOfing/altered/altered_medium/')
altered_hard_images_path, altered_hard = load_data('SOCOfing/altered/altered_hard/')

preprocessed_real = [preprocess_image(img) for img in real_images]
preprocessed_altered_easy = [preprocess_image(img) for img in altered_easy]
preprocessed_altered_medium = [preprocess_image(img) for img in altered_medium]
preprocessed_altered_hard = [preprocess_image(img) for img in altered_hard]

scores = calculating_matches_between_images(altered_easy_images_path, preprocessed_altered_easy, altered_medium_images_path, preprocessed_altered_medium)
overall_score = np.mean(scores)

print(f'Overall score - {overall_score}')

# matching of two fingerprints for testing

# img1 = cv2.imread('SOCOfing/real/1__M_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('SOCOfing/altered/altered_easy/1__M_Left_index_finger_Obl.BMP', cv2.IMREAD_GRAYSCALE)

# img_test = preprocess_image(img1)
# img_test2 = preprocess_image(img2)

# matches = detect_and_match_fingers(img_test, img_test2)
# good_matches = calculate_matches(matches)

# print(len(good_matches)/len(matches))

