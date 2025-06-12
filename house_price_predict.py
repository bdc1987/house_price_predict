"""
Dự đoán giá nhà Hà Nội bằng Random Forest
- Tiền xử lý dữ liệu
- Huấn luyện mô hình
- Lưu mô hình và encoder
- Dự đoán giá nhà mới với input dạng tên (không cần số hóa thủ công)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 1. Đọc dữ liệu
df = pd.read_csv("HN_Houseprice_new.csv")

# 2. Xử lý cột Price
def convert_price(val):
    if isinstance(val, str):
        val = val.replace(",", ".").strip()
        if "tỷ" in val:
            try:
                return float(val.replace("tỷ", "").strip()) * 1e9
            except:
                return np.nan
        elif "triệu" in val:
            try:
                return float(val.replace("triệu", "").strip()) * 1e6
            except:
                return np.nan
    return np.nan

df["Price_num"] = df["Price"].apply(convert_price)

# 3. Xử lý cột Area
df["Area_num"] = pd.to_numeric(
    df["Area"].str.extract(r'(\d+[\.,]?\d*)')[0].str.replace(",", "."),
    errors='coerce'
)

# 4. Trích số tầng, mặt tiền
df["Floors_num"] = pd.to_numeric(
    df["Floors"].astype(str).str.extract(r'(\d+)')[0],
    errors='coerce'
)
df["Width_meters_num"] = pd.to_numeric(
    df["Width_meters"].astype(str).str.extract(r'(\d+[\.,]?\d*)')[0].str.replace(",", "."),
    errors='coerce'
)

# 5. Lọc các cột cần thiết
input_cols = [
    "Area_num", "District", "PostType", "Bedrooms", "Bathrooms",
    "Floors_num", "Direction", "Width_meters_num"
]
target_col = "Price_num"
df_model = df[input_cols + [target_col]].copy()

# 6. Chuẩn hóa cột Bedrooms và Bathrooms về dạng số
df_model["Bedrooms"] = pd.to_numeric(
    df_model["Bedrooms"].astype(str).str.extract(r'(\d+)')[0],
    errors='coerce'
)
df_model["Bathrooms"] = pd.to_numeric(
    df_model["Bathrooms"].astype(str).str.extract(r'(\d+)')[0],
    errors='coerce'
)

# 7. Mã hóa các cột dạng chữ và lưu encoder
label_cols = ["District", "PostType", "Direction"]
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

# 8. Loại bỏ các dòng thiếu dữ liệu
df_model = df_model.dropna()

# 9. Chia tập train/test
train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)

# 10. Huấn luyện mô hình Random Forest
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R2 score:", r2)

# 12. Lưu mô hình, cột đầu vào, encoder
joblib.dump(model, "house_price_rf_model.pkl")
joblib.dump(list(X_train.columns), "house_price_features.pkl")
joblib.dump(encoders, "house_price_encoders.pkl")

# 13. Hàm dự đoán giá nhà mới
def predict_house_price(input_dict):
    model = joblib.load("house_price_rf_model.pkl")
    feature_names = joblib.load("house_price_features.pkl")
    encoders = joblib.load("house_price_encoders.pkl")
    input_df = pd.DataFrame([input_dict])
    # Mã hóa các cột dạng chữ
    for col in ["District", "PostType", "Direction"]:
        le = encoders[col]
        input_df[col] = le.transform([input_df[col][0]])
    input_df = input_df[feature_names]
    price = model.predict(input_df)[0]
    return price

# 14. Ví dụ sử dụng
if __name__ == "__main__":
    # Thay đổi các giá trị bên dưới cho phù hợp với căn nhà bạn muốn dự đoán
    input_dict = {
        "Area_num": 90,
        "District": "Ba Đình",      # Gõ đúng tên quận/huyện như trong dữ liệu gốc
        "PostType": "Bán nhà riêng",# Gõ đúng loại hình như trong dữ liệu gốc
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Floors_num": 4,
        "Direction": "Đông",        # Gõ đúng hướng như trong dữ liệu gốc
        "Width_meters_num": 4.5
    }
    predicted_price = predict_house_price(input_dict)
    print(f"Giá nhà dự đoán: {predicted_price:,.0f} VND")
