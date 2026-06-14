import os
import pandas as pd
import yaml
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def train_models():
    print("Đang tải dữ liệu...")
    train_df = pd.read_csv(config['data']['train_path'])
    test_df = pd.read_csv(config['data']['test_path'])

    # Tách Features (X) và Labels (y)
    X_train = train_df.drop(columns=['Rating', 'Recommended IND'])
    y_train_reg = train_df['Rating']             # Cho Hồi quy
    y_train_clf = train_df['Recommended IND']    # Cho Phân loại

    X_test = test_df.drop(columns=['Rating', 'Recommended IND'])
    y_test_reg = test_df['Rating']
    y_test_clf = test_df['Recommended IND']

    # 2. Bài toán Hồi quy: Dự đoán Rating (1-5 sao)
    print("Đang huấn luyện Linear Regression...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train_reg)

    # 3. Bài toán Phân loại: Dự đoán khả năng khuyên dùng (0 hoặc 1)
    print("Đang huấn luyện Logistic Regression...")
    log_reg = LogisticRegression(
        C=config['model']['logistic_regression']['C'],
        max_iter=config['model']['logistic_regression']['max_iter'],
        random_state=config['model']['random_state']
    )
    log_reg.fit(X_train, y_train_clf)

    # 4. Đánh giá nhanh trên tập Test
    reg_preds = lin_reg.predict(X_test)
    clf_preds = log_reg.predict(X_test)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Linear Regression MSE: {mean_squared_error(y_test_reg, reg_preds):.4f}")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test_clf, clf_preds):.4f}")

    # 5. Lưu mô hình
    os.makedirs("models", exist_ok=True)
    joblib.dump(lin_reg, "models/linear_model.pkl")
    joblib.dump(log_reg, "models/logistic_model.pkl")
    print("\nĐã lưu mô hình vào thư mục models/")

if __name__ == "__main__":
    train_models()