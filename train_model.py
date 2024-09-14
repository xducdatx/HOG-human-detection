import numpy as np
import cv2  
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import glob
import joblib
from my_lib import model

def conv_sobel (matrix, kernel):
    height, width = matrix.shape
    k_height, k_width = kernel.shape
    new_matrix = np.zeros((8, 8), dtype=np.float64)
    for i in range(height - k_height + 1):
        for j in range(width - k_width + 1):
            block = matrix[i:i+k_height, j:j+k_width]
            new_matrix[i, j] = np.sum(block * kernel)
    return new_matrix

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
            # orientation_value = orientation[i, j] % 180
            # bin_index1 = int(orientation_value / bins_width)
            # bin_index2 = (bin_index1 + 1)
            # histogram[bin_index1] += magnitude[i, j] * (bin_index2 * bins_width - orientation_value) / bins_width
            # histogram[bin_index2 % nbins] += magnitude[i, j] * (orientation_value - bin_index1 * bins_width) / bins_width
            histogram[int(orientation[i, j] / bins_width) % nbins] += magnitude[i, j]
    return histogram

def l2_normalize(vector, epsilon=1e-6):
    l2_norm = np.sqrt(np.sum(vector ** 2) + epsilon)
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

count_greater_than_511 = 0
max_val = 0
min_val = 0
def hog(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    global count_greater_than_511
    global max_val
    global min_val
    #Padding zero
    # padded_image = np.pad(gray_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    height, width = image.shape
    gx = np.zeros((height, width), dtype=np.float32)
    gy = np.zeros((height, width), dtype=np.float32)
    block_size = 8
    all_histograms = []
    print("still running")
    for i in range(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            #update blocksize * 2 o block line 54
            block_16x16 = image[i:i+block_size * 2 , j:j+block_size *2]
            block_8x8_1 = block_16x16[0:8, 0:8]
            block_8x8_2 = block_16x16[0:8, 8:16]
            block_8x8_3 = block_16x16[8:16, 0:8] 
            block_8x8_4 = block_16x16[8:16, 8:16] 

            padded_image = np.pad(block_8x8_1, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(padded_image)
            gy = sobel_extraction_Gy(padded_image)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_1 = compute_histogram(magnitude, orientation)
            # count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            # max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))

            padded_image = np.pad(block_8x8_2, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(padded_image)
            gy = sobel_extraction_Gy(padded_image)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_2 = compute_histogram(magnitude, orientation)
            # count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            # max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))

            padded_image = np.pad(block_8x8_3, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(padded_image)
            gy = sobel_extraction_Gy(padded_image)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_3 = compute_histogram(magnitude, orientation)
            # count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            # max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))

            padded_image = np.pad(block_8x8_4, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # gx = conv_sobel(padded_image, sobel_x)
            # gy = conv_sobel(padded_image, sobel_y)
            gx = sobel_extraction_Gx(padded_image)
            gy = sobel_extraction_Gy(padded_image)
            magnitude, orientation = compute_gx_gy(gx, gy)
            histogram_4 = compute_histogram(magnitude, orientation)
            # count_greater_than_511 += np.sum(gx > 511) + np.sum(gy > 511) + np.sum(magnitude > 511) + np.sum(orientation > 511)
            # max_val = max(max_val, np.max(gx), np.max(gy), np.max(magnitude), np.max(orientation))
            # min_val = min(min_val, np.min(gx), np.min(gy), np.min(magnitude), np.min(orientation))

            combined_histogram = np.concatenate((histogram_1, histogram_2, histogram_3, histogram_4))
            # count_greater_than_511 += np.sum(combined_histogram > 511)
            # max_val = max(max_val, np.max(combined_histogram))
            # min_val = min(min_val, np.min(combined_histogram))
            normalize_histogram = l2_normalize(combined_histogram)
            all_histograms.extend(normalize_histogram)

    all_histograms = np.array(all_histograms)
    return all_histograms
# print(all_histograms.shape)
def resize_inter_area(img_array, new_height, new_width):
    height, width = img_array.shape
    x_ratio = width / new_width
    y_ratio = height / new_height

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
    height, width = image.shape
    print("height", height)
    print("width", width)

    for i in range(0, height - window_size[0], step_size[0]):
        for j in range(0, width - window_size[1], step_size[1]):
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
    return windows

def main():
    #-----------------------------------------------------
    # train test
    global count_greater_than_511 
    global max_val
    global min_val
    image_paths_pos = glob.glob('pos_images_train_2/*')
    image_paths_neg = glob.glob('neg_images_train_2/*')

    hog_features_list = []
    labels = [1] * 3542 + [0] * 9744
    for image_path in image_paths_pos:
        image = cv2.imread(image_path)
        gray_image = img_to_gray(image)
        hog_features = hog(gray_image)
        hog_features_list.append(hog_features)

    for image_path in image_paths_neg:
        image = cv2.imread(image_path)
        gray_image = img_to_gray(image)
        hog_features = hog(gray_image)
        hog_features_list.append(hog_features)

    X = np.array(hog_features_list)
    y = np.array(labels)
    print(X.shape)
    print(X)
    print(y.shape)
    print(y)
    # #-----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear'],
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
    grid.fit(X_train, y_train)  
    print("Best parameters found: ", grid.best_params_)
    best_model = grid.best_estimator_


    # Huấn luyện mô hình SVM

#######


    # model = SVC(kernel='linear')
    # model.fit(X_train, y_train)
#######   
    # Dự đoán trên tập kiểm tra
    y_pred = best_model.predict(X_test)
    
    # Đánh giá độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Predicted labels:", y_pred)
    print("True labels:", y_test)
    
    # In ra các tham số của mô hình
    print("Model parameters:", best_model.get_params())
    if best_model.kernel == 'linear':
        print("Model coefficients shape:", best_model.coef_.shape)
        print("Model coefficients:", best_model.coef_)
    joblib.dump(best_model, 'svm_model_14-9_gridS.pkl')
    print("Model saved to svm_model_14-9_gridS.pkl")

    print("count_greater_than_511", count_greater_than_511)
    print("max_val", max_val)
    print("min_val", min_val)
if __name__ == "__main__":
    main()



