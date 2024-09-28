import numpy as np



import numpy as np
import cv2  
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import glob

def sobel_extraction_Gx (matrix):
    new_matrix = np.zeros((8, 8), dtype=np.float64)
    matrix = matrix.astype(np.float64)
    for i in range(8):
        for j in range(8):
            new_matrix[i, j] = matrix[i + 2, j + 1] - matrix[i, j + 1]

    return new_matrix

def sobel_extraction_Gy (matrix):
    new_matrix = np.zeros((8, 8), dtype=np.float64)
    matrix = matrix.astype(np.float64)
    for i in range(8):
        for j in range(8):
            new_matrix[i, j] = matrix[i + 1, j + 2] - matrix[i + 1, j]

    return new_matrix

def compute_gx_gy(gx, gy):
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi)  #radients to degrees
    # Convert negative angles
    orientation = np.where(orientation < 0, orientation + 180, orientation)
    return magnitude, orientation

def compute_histogram(magnitude, orientation, nbins = 9):
    bins_width = 180 / nbins
    histogram = np.zeros(nbins)
    # print(histogram)
    for i in range(8):
        for j in range(8):
            histogram[int(orientation[i, j] / bins_width) % nbins] += magnitude[i, j]
    return histogram

count_greater_than_511 = 0
max_val = 0
min_val = 100

def l2_normalize(vector, epsilon=1e-6):
    global max_val
    global min_val
    l2_norm = np.sqrt(np.sum(vector ** 2) + epsilon)
    max_val = max(max_val, np.sum(vector ** 2))
    normalized_vector = vector / l2_norm
    return normalized_vector

def img_to_gray(image):
    height, width, _ = image.shape
    gray_image = np.zeros((height, width), dtype=np.uint8)  
    for i in range(height):
        for j in range(width):
            B, G, R = image[i, j]
            gray_value = 0.299 * R + 0.587 * G + 0.114 * B
            gray_image[i, j] = int(gray_value)
    return gray_image

def hog(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    global count_greater_than_511
    global max_val
    global min_val
    #Padding zero
    image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    height, width = image.shape
    gx = np.zeros((height, width), dtype=np.float32)
    gy = np.zeros((height, width), dtype=np.float32)
    block_size = 8
    all_histograms = []
    print("still running")
    for i in range(0, height - block_size - 2, block_size):
        for j in range(0, width - block_size - 2, block_size):
            #update blocksize * 2 o block line 54
            # print(i, j)
            block_16x16 = image[i:i+block_size * 2 + 2 , j:j+block_size *2 + 2]
            block_8x8_1 = block_16x16[0:10, 0:10]
            block_8x8_2 = block_16x16[0:10, 8:18]
            block_8x8_3 = block_16x16[8:18, 0:10] 
            block_8x8_4 = block_16x16[8:18, 8:18] 

            # padded_image = np.pad(block_8x8_1, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_1)
            gy = sobel_extraction_Gy(block_8x8_1)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_1 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))


            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            # padded_image = np.pad(block_8x8_2, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_2)
            gy = sobel_extraction_Gy(block_8x8_2)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_2 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))
            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            # padded_image = np.pad(block_8x8_3, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_3)
            gy = sobel_extraction_Gy(block_8x8_3)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_3 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))
            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            # padded_image = np.pad(block_8x8_4, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_4)
            gy = sobel_extraction_Gy(block_8x8_4)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_4 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))
            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            combined_histogram = np.concatenate((histogram_1, histogram_2, histogram_3, histogram_4))
            count_greater_than_511 += np.sum(combined_histogram > 511)
            max_val = max(max_val, np.max(combined_histogram))
            if np.any(combined_histogram > 0):
                min_val = min(min_val, np.min(combined_histogram[combined_histogram > 0])) 
            # min_val = min(min_val, np.min(combined_histogram[combined_histogram > 0])) 
            normalize_histogram = l2_normalize(combined_histogram)
            all_histograms.extend(normalize_histogram)

    all_histograms = np.array(all_histograms)
    return all_histograms



def hog_2(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    global count_greater_than_511
    global max_val
    global min_val
    #Padding zero
    image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    height, width = image.shape
    gx = np.zeros((height, width), dtype=np.float32)
    gy = np.zeros((height, width), dtype=np.float32)
    block_size = 8
    all_histograms = []
    print(image.shape)
    print("still running")
    for i in range(1, height - block_size - 1, block_size):
        for j in range(1, width - block_size - 1, block_size):
            # Lấy block 16x16 từ ma trận đã padding
            # print(i, j)
            block_16x16 = image[i-1:i+block_size*2+1, j-1:j+block_size*2+1]
            block_8x8_1 = block_16x16[0:10, 0:10]
            block_8x8_2 = block_16x16[0:10, 8:18]
            block_8x8_3 = block_16x16[8:18, 0:10] 
            block_8x8_4 = block_16x16[8:18, 8:18] 

            # padded_image = np.pad(block_8x8_1, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_1)
            gy = sobel_extraction_Gy(block_8x8_1)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_1 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))


            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            # padded_image = np.pad(block_8x8_2, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_2)
            gy = sobel_extraction_Gy(block_8x8_2)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_2 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))
            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            # padded_image = np.pad(block_8x8_3, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_3)
            gy = sobel_extraction_Gy(block_8x8_3)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_3 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))
            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            # padded_image = np.pad(block_8x8_4, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(block_8x8_4)
            gy = sobel_extraction_Gy(block_8x8_4)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_4 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx[gx > 0]), np.min(gy[gy > 0]), np.min(magnitude[magnitude > 0]), np.min(orientation[orientation > 0]))
            if np.any(gx > 0):
                min_val = min(min_val, np.min(gx[gx > 0]))
            if np.any(gy > 0):
                min_val = min(min_val, np.min(gy[gy > 0]))
            if np.any(magnitude > 0):
                min_val = min(min_val, np.min(magnitude[magnitude > 0]))
            if np.any(orientation > 0):
                min_val = min(min_val, np.min(orientation[orientation > 0]))

            combined_histogram = np.concatenate((histogram_1, histogram_2, histogram_3, histogram_4))
            count_greater_than_511 += np.sum(combined_histogram > 511)
            max_val = max(max_val, np.max(combined_histogram))
            if np.any(combined_histogram > 0):
                min_val = min(min_val, np.min(combined_histogram[combined_histogram > 0])) 
            # min_val = min(min_val, np.min(combined_histogram[combined_histogram > 0])) 
            normalize_histogram = l2_normalize(combined_histogram)
            all_histograms.extend(normalize_histogram)

    all_histograms = np.array(all_histograms)
    return all_histograms





def create_block_matrix():
    # Khởi tạo ma trận zeros với kích thước 128x64
    matrix = np.zeros((128, 64), dtype=np.int32)
    
    # Điền các giá trị vào các block 8x8
    for i in range(0, 128, 8):
        for j in range(0, 64, 8):
            block_value = (i // 8) * (64 // 8) + (j // 8) + 1  # Giá trị block từ 1 đến 128
            matrix[i:i+8, j:j+8] = block_value
    
    return matrix

# Tạo ma trận và in ra để kiểm tra
# np.set_printoptions(threshold=np.inf)
block_matrix = create_block_matrix()
print("Original Block Matrix:")
# print(block_matrix)
print(block_matrix.shape)
hog_feature =  hog(block_matrix)
print(hog_feature.shape)
print(hog_feature)
hog_feature_2 = hog_2(block_matrix)
print(hog_feature_2.shape)
print(hog_feature_2)
# # Padding ma trận
# padded_matrix = np.pad(block_matrix, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# print("Padded Matrix:")
# print(padded_matrix)
# print(padded_matrix.shape)

# # Lấy các block 16x16 và chia thành các cell 10x10
# height, width = padded_matrix.shape
# block_size = 8
# for i in range(1, height - block_size, block_size):
#     for j in range(1, width - block_size, block_size):
#         # Lấy block 16x16 từ ma trận đã padding
#         block_16x16 = padded_matrix[i-1:i+block_size*2+1, j-1:j+block_size*2+1]
        
#         # Chia block 16x16 thành 4 cell 10x10
#         block_8x8_1 = block_16x16[0:10, 0:10]
#         block_8x8_2 = block_16x16[0:10, 8:18]
#         block_8x8_3 = block_16x16[8:18, 0:10]
#         block_8x8_4 = block_16x16[8:18, 8:18]

#         print("Block 16x16:")
#         print(block_16x16)
#         print("Block 8x8 1 (10x10):")
#         print(block_8x8_1)
#         print("Block 8x8 2 (10x10):")
#         print(block_8x8_2)
#         print("Block 8x8 3 (10x10):")
#         print(block_8x8_3)
#         print("Block 8x8 4 (10x10):")
#         print(block_8x8_4)


