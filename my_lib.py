import joblib
import numpy as np
# Tải mô hình từ tệp .pkl
model = joblib.load('svm_model.pkl')
a = np.min(model.coef_[model.coef_ > 0])
print("-------MIN--------")
print (a)
print("------------------")
# In ra mô hình đã tải
print("Loaded model:", model)
print (model.coef_)
print("-------MAX--------")
print(np.max(model.coef_))
print("------------------a")




# Kiểm tra nếu mô hình sử dụng kernel tuyến tính và in ra các hệ số
if hasattr(model, 'coef_'):
    print("Model coefficients shape:", model.coef_.shape)
    print("Model coefficients:", model.coef_)
    print("Model bias:", model.intercept_)
else:
    print("The model does not have coefficients (it may not be a linear SVM model).")

# print(np.min(np.abs(model.coef_)[model.coef_ > 0]))
# print(np.max(np.abs(model.coef_)))

def real_to_fixed_bitstring(real_num):
    # Định nghĩa số bit cho phần integer và fraction
    int_bits = 4
    frac_bits = 28
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
    
    # Chuyển số fixed-point thành chuỗi bit
    bitstring = format(fixed_num, f'0{total_bits}b')
    return bitstring

# Ví dụ sử dụng
real_num = 3.1415926535
bitstring = real_to_fixed_bitstring(real_num)
print(f"Fixed-point bitstring: {bitstring}")

with open('coefficients_fixed_point.txt', 'w') as file:
    # Duyệt qua tất cả các hệ số trong model.coef_
    for idx, coef in enumerate(model.coef_.flatten(), start=1):
        # Chuyển đổi hệ số thành bit fixed-point
        bitstring = real_to_fixed_bitstring(coef)
        # Ghi số thứ tự và bitstring vào tệp
        file.write(f"{idx}: {bitstring}\n")

print("Đã lưu các hệ số fixed-point vào tệp 'coefficients_fixed_point.txt'")