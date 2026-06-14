import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    input_path = "data/raw/Womens Clothing E-Commerce Reviews.csv"
    print(f"Read raw data...")
    df = pd.read_csv(input_path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df.dropna(subset=['Review Text', 'Division Name', 'Department Name', 'Class Name'])

    df['Review Length'] = df['Review Text'].apply(len)

    numeric_features = df[['Review Length', 'Age', 'Positive Feedback Count']]

    categorical_features = pd.get_dummies(df[['Division Name', 'Department Name', 'Class Name']], drop_first=True)

    X = pd.concat([numeric_features, categorical_features], axis=1)

    # Nhãn cho 2 bài toán (Rating để Hồi quy, Recommended IND để Phân loại)
    y_reg = df['Rating']
    y_clf = df['Recommended IND']

    processed_df = pd.concat([X, y_reg, y_clf], axis=1)

    print("Đang chia tập Train/Test...")
    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"Hoàn tất! Kích thước tập Train: {train_df.shape}, tập Test: {test_df.shape}")

if __name__ == "__main__":
    preprocess_data()