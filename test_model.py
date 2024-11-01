import numpy as np
import cv2  
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import glob
import joblib
from my_lib import model
import time

def sobel_extraction_Gy (matrix):
    new_matrix = np.zeros((8, 8), dtype=np.float64)
    matrix = matrix.astype(np.float64)
    for i in range(8):
        for j in range(8):
            new_matrix[i, j] = matrix[i + 2, j + 1] - matrix[i, j + 1]
    return new_matrix

def sobel_extraction_Gx (matrix):
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
    vector = vector / (np.sum(vector) + epsilon)
    vector = np.sqrt(vector)
    max_val = max(max_val, np.sum(vector))
    return vector

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

            block_16x16 = image[i:i+block_size * 2 +2, j:j+block_size *2+2]
            block_8x8_1 = block_16x16[0:10, 0:10]
            block_8x8_2 = block_16x16[0:10, 8:18]
            block_8x8_3 = block_16x16[8:18, 0:10] 
            block_8x8_4 = block_16x16[8:18, 8:18] 


            gx = sobel_extraction_Gx(block_8x8_1)
            gy = sobel_extraction_Gy(block_8x8_1)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_1 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))


            gx = sobel_extraction_Gx(block_8x8_2)
            gy = sobel_extraction_Gy(block_8x8_2)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_2 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))


            gx = sobel_extraction_Gx(block_8x8_3)
            gy = sobel_extraction_Gy(block_8x8_3)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_3 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))


            gx = sobel_extraction_Gx(block_8x8_4)
            gy = sobel_extraction_Gy(block_8x8_4)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_4 = compute_histogram(magnitude, orientation)
            count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))

            combined_histogram = np.concatenate((histogram_1, histogram_2, histogram_3, histogram_4))
            count_greater_than_511 += np.sum(combined_histogram > 511)
            max_val = max(max_val, np.max(combined_histogram))
            min_val = min(min_val, np.min(combined_histogram))
            normalize_histogram = l2_normalize(combined_histogram)
            all_histograms.extend(normalize_histogram)

    all_histograms = np.array(all_histograms)
    return all_histograms

# print(all_histograms.shape)
def resize_inter_area(img_array, new_height, new_width):
    height, width = img_array.shape
    x_ratio = width / new_width
    y_ratio = height /new_height

    resized_array = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            x_start = int(j * x_ratio)
            x_end = min(int((j + 1) * x_ratio), width)
            y_start = int(i * y_ratio)
            y_end = min(int((i + 1) * y_ratio), height)

            # Tính giá trị trung bình cho điểm ảnh mới
            block = img_array[y_start:y_end, x_start:x_end]
            resized_array[i, j] = np.max(block)

    return resized_array.astype(np.uint8)

def sliding_windows(image, window_size, step_size):
    windows = []
    windows_index = []
    height, width = image.shape
    print("height", height)
    print("width", width)

    for i in range(0, height - window_size[0] + 1, step_size[0]):
        for j in range(0, width - window_size[1] + 1, step_size[1]):
            if (i + window_size[0] > height) and (j + window_size[1] > width):
                window = image[height - window_size[0] : height, width - window_size[1] : width]
                print("Out of height & width")
                print("i: ", height - window_size[0], "-> ", height)
                print("j: ", width - window_size[1],  "-> ", width)
            elif (i + window_size[0] > height):
                window = image[height - window_size[0] : height, j : j + window_size[1]]
                print("Out of height")
                print("i: ", height - window_size[0], "-> ", height)
                print("j: ", j, "-> ", j + window_size[1])
            elif (j + window_size[1] > width):
                window = image[i : i + window_size[0], width - window_size[1] : width]
                print("Out of width")
                print("i: ", i , "-> ", i + window_size[0])
                print("j: ", width - window_size[1],  "-> ", width)
            else:
                window = image[i : i + window_size[0], j : j + window_size[1]]
                print("i: ", i , "-> ", i + window_size[0])
                print("j: ", j,  "-> ", j + window_size[1])
            print("\n")
            windows.append(window)
            windows_index.append((i, j, i + window_size[0], j + window_size[1]))
    return windows, windows_index

def draw_boundaries(image, positions):
    for (x1, y1, x2, y2) in positions:
        cv2.rectangle(image, (y1, x1), (y2, x2), (0, 255, 0), 2)
    return image

def main():
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Không thể mở camera")
    #     return
    # while cap.isOpened():
    #     # Đọc khung hình từ camera
    #     ret, frame = cap.read()
        
    #     if not ret:
    #         print("Không thể đọc khung hình từ camera")
    #         break

    #     # Chuyển đổi sang ảnh grayscale
    #     gray_image = img_to_gray(frame)

    #     # Resize hoặc không tùy thuộc vào yêu cầu của bạn
    #     resize_image = resize_inter_area(gray_image, 240, 320)   # (height, width)
    #     # resize_image = gray_image

    #     # Sử dụng sliding windows để quét qua ảnh
    #     window_size = (128, 64) # (height, width)
    #     step_size = (128, 64)
    #     windows, windows_index = sliding_windows(resize_image, window_size, step_size)

    #     count_frame_people = 0
    #     count_frame_people_2 = 0

    #     # Duyệt qua các cửa sổ và áp dụng HOG
    #     for idx, window in enumerate(windows):
    #         print("Window shape: ", window.shape)
            
    #         # Tính các đặc trưng HOG cho cửa sổ hiện tại
    #         hog_features = hog(window)
    #         hog_features_reshape = hog_features.reshape(1, -1)

    #         # Dự đoán bằng model SVM của bạn
    #         if model.predict(hog_features_reshape) == 1:
    #             count_frame_people += 1
    #             print("idx: ", idx)
    #             resize_image = draw_boundaries(resize_image, [windows_index[idx]])

    #         # Cách tính dựa vào coef_ và intercept_
    #         if (np.sum(hog_features_reshape * model.coef_) + model.intercept_ > 0):
    #             count_frame_people_2 += 1

    #     # Hiển thị khung hình đã phát hiện
    #     cv2.imshow('Detected Image', resize_image)

    #     # Nhấn 'q' để thoát
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Giải phóng camera và đóng các cửa sổ hiển thị
    # cap.release()
    # cv2.destroyAllWindows()

    image = cv2.imread('640x480/110153.jpg')
    start_time = time.time();
    gray_image = img_to_gray(image)
    window_size = (128, 64) # (height, width)
    step_size = (16, 8)

    global count_greater_than_511 
    global max_val
    global min_val
    
    max_cal_detect = 0

    count_frame_people_2 = 0
    resize_image = resize_inter_area(gray_image, 240, 320)   # (height, width)
    # resize_image = gray_image
    windows, windows_index = sliding_windows(resize_image, window_size, step_size)
    print("Number of windows: ", len(windows))
    count_frame_people = 0
    for idx, window in enumerate(windows):

        print("Window shape: ", window.shape)
        
        hog_features = hog(window)
        hog_features_reshape = hog_features.reshape(1, -1)
        if model.predict(hog_features_reshape) == 1:
            count_frame_people += 1
            print("idx: ", idx)
            resize_image = draw_boundaries(resize_image, [windows_index[idx]])
        if (np.sum( hog_features_reshape *  model.coef_) + model.intercept_ > 0):
            if max_cal_detect < np.sum( hog_features_reshape *  model.coef_) + model.intercept_: 
                max_cal_detect = np.sum( hog_features_reshape *  model.coef_) + model.intercept_
            count_frame_people_2 += 1
    end_time = time.time()
    print("Time: ", end_time - start_time)
    # for window in windows_index:
    #     print(window)
    cv2.imshow('Detected Image', resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("max_cal_detect: ", max_cal_detect)


    # print("Count frame people: ", count_frame_people)
    # print("Count frame people 2 : ", count_frame_people_2)
    # print("Count greater than 511: ", count_greater_than_511)
    # print("Max value: ", max_val)
    # print("Min value: ", min_val)
    # print("bias: ", model.intercept_)


if __name__ == "__main__":
    main()


