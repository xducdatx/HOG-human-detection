import joblib
import numpy as np

# Tải mô hình từ tệp .pkl
model = joblib.load('svm_model_3_3.pkl') ##old model
# model = joblib.load('svm_model_26-2-new.pkl') ##new_model 26/2/2025
# model = joblib.load('svm_model_30-10.pkl')
# Tìm giá trị nhỏ nhất trong model.coef_ mà lớn hơn 0
a = np.min(model.coef_[model.coef_ > 0])
print("-------MIN--------")
print(a)
print("------------------")

# In ra mô hình đã tải
print("Loaded model:", model)
print(model.coef_)
print("-------MAX--------")
print(np.max(model.coef_))
print("------------------")

# Kiểm tra nếu mô hình sử dụng kernel tuyến tính và in ra các hệ số
if hasattr(model, 'coef_'):
    print("Model coefficients shape:", model.coef_.shape)
    print("Model coefficients:", model.coef_)
    print("Model bias:", model.intercept_)
else:
    print("The model does not have coefficients (it may not be a linear SVM model).")

def real_to_fixed_hex(real_num, int_bits=4, frac_bits=12):
    total_bits = int_bits + frac_bits
    
    # Tính hệ số nhân để chuyển đổi phần fraction
    scale_factor = 1 << frac_bits
    
    # Chuyển số thực thành số fixed-point
    fixed_num = int(real_num * scale_factor)
    
    # Kiểm tra overflow và điều chỉnh trong giới hạn bit integer
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    if fixed_num > max_val:
        fixed_num = max_val
    elif fixed_num < min_val:
        fixed_num = min_val
    
    # Nếu số âm, chuyển về số dương hai bù (two's complement)
    if fixed_num < 0:
        fixed_num = (1 << total_bits) + fixed_num  # Two's complement
    
    # Chuyển số fixed-point thành chuỗi hex
    hex_string = format(fixed_num, f'0{total_bits // 4}X')
    return hex_string
# print("------------------")
# print (real_to_fixed_hex(1e-6))
# print("------------------")
print("HEHE")
print(real_to_fixed_hex(0.016))

with open('coefficients_fixed_point_old.txt', 'w') as file:
    # Duyệt qua tất cả các hệ số trong model.coef_
    coefficients = model.coef_.flatten()
    num_coefficients = len(coefficients)
    
    # Chia các hệ số thành các cụm 9 phần tử
    for i in range(0, num_coefficients, 9):
        cluster = coefficients[i:i+9]
        # Sắp xếp lại thứ tự trong cụm (thứ tự ngược lại)
        cluster = cluster[::-1]
        
        # Chuyển đổi từng hệ số trong cụm thành hex fixed-point và ghi vào tệp
        hex_strings = [real_to_fixed_hex(coef) for coef in cluster]
        file.write(''.join(hex_strings) + '\n')

print("Đã lưu các hệ số fixed-point dưới dạng hex vào tệp 'coefficients_fixed_point_old.txt'")



coefficients = model.coef_.flatten()
num_coefficients = len(coefficients)

# Định dạng lại các hệ số theo kiểu ma trận 105 hàng và 36 cột
matrix = np.zeros((36, 105), dtype=object)
for i in range(36):
    for j in range(105):
        index = 3744 + i - j * 36
        if index < num_coefficients:
            matrix[i, j] = real_to_fixed_hex(coefficients[index])

# Ghi các hệ số vào tệp
with open('coefficients_fixed_point.txt', 'w') as file:
    for row in matrix:
        file.write(''.join(row) + '\n')

print("Đã lưu các hệ số fixed-point dưới dạng hex vào tệp 'coefficients_fixed_point.txt'")


print(real_to_fixed_hex(coefficients[3778]))