import joblib
import numpy as np
# Tải mô hình từ tệp .pkl
model = joblib.load('svm_model_15-9_gridS_2.pkl')
a = np.min(model.coef_[model.coef_ > 0])
print("------------------")
print (a)
print("------------------")
# In ra mô hình đã tải
print("Loaded model:", model)
print (model.coef_)
# Kiểm tra nếu mô hình sử dụng kernel tuyến tính và in ra các hệ số
if hasattr(model, 'coef_'):
    print("Model coefficients shape:", model.coef_.shape)
    print("Model coefficients:", model.coef_)
    print("Model bias:", model.intercept_)
else:
    print("The model does not have coefficients (it may not be a linear SVM model).")
