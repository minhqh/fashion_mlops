# Cấu trúc dự án

``` fashion_mlops
fashion_mlops/
├── .github/              # Nơi chứa các file workflow (CI/CD) của GitHub Actions sau này
├── configs/              # Nơi chứa config.yaml (hyperparameters, đường dẫn data)
├── data/                 # Thư mục chứa dữ liệu 
│   ├── raw/              # Dữ liệu gốc kéo từ Kaggle về
│   └── processed/        # Dữ liệu đã qua làm sạch, chuẩn hóa
├── models/               # Nơi lưu trữ các model đã train (vd: linear_model.pkl)
├── notebooks/            # Jupyter notebooks chỉ dùng để nháp, visualize data (EDA)
├── src/                  # Thư mục lõi chứa source code
│   ├── __init__.py
│   ├── get_data.py       # Script gọi Kaggle API để tải data
│   ├── preprocess.py     # Script xử lý missing values, encode text
│   ├── train.py          # Script chứa model Linear & Logistic Regression
│   └── evaluate.py       # Tính toán accuracy, MSE...
├── .gitignore            # Cực kỳ quan trọng: Chặn push data thật và file nhạy cảm
├── requirements.txt      # Khai báo các thư viện cần thiết
└── README.md             # Giới thiệu sơ lược về project
```
