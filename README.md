# 🚗 Helmet Detection - Nhận Dạng Mũ Bảo Hiểm

Dự án nhận dạng và phân loại người có đội mũ bảo hiểm hay không sử dụng Deep Learning và Computer Vision. Dự án sử dụng Transfer Learning với MobileNetV2 để đạt được độ chính xác cao trong việc phân loại hình ảnh.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Mục Lục

- [Tính Năng](#-tính-năng)
- [Dataset](#-dataset)
- [Mô Hình](#-mô-hình)
- [Trích Chọn Đặc Trưng](#-trích-chọn-đặc-trưng)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt](#-cài-đặt)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Training](#-training)
- [Kết Quả](#-kết-quả)
- [Demo/Inference](#-demoinference)
- [Tác Giả](#-tác-giả)

## ✨ Tính Năng

- ✅ **Phân loại nhị phân**: Nhận dạng người có đội mũ bảo hiểm hay không
- ✅ **Transfer Learning**: Sử dụng MobileNetV2 pre-trained trên ImageNet
- ✅ **Data Augmentation**: Tăng cường dữ liệu để cải thiện hiệu suất model
- ✅ **High Accuracy**: Đạt 100% accuracy trên test set
- ✅ **Real-time Detection**: Hỗ trợ nhận dạng real-time qua camera
- ✅ **Batch Processing**: Xử lý nhiều ảnh cùng lúc
- ✅ **Visualization**: Hiển thị kết quả với đồ thị và confusion matrix

## 📊 Dataset

### Thông Tin Dataset

- **Tổng số ảnh**: 149 ảnh
- **Số lớp**: 2 (no_helmet, with_helmet)
- **Định dạng**: JPG, PNG, WEBP
- **Kích thước ảnh**: Được resize về 224x224 pixels

### Phân Chia Dataset

```
Dataset được chia theo tỷ lệ: 70% Train / 15% Validation / 15% Test

Train:   103 ảnh
  ├── no_helmet:    54 ảnh
  └── with_helmet:   49 ảnh

Validation: 21 ảnh
  ├── no_helmet:    11 ảnh
  └── with_helmet:   10 ảnh

Test: 25 ảnh
  ├── no_helmet:    13 ảnh
  └── with_helmet:   12 ảnh
```

### Cấu Trúc Dataset

```
dataset/
├── train/
│   ├── no_helmet/
│   │   └── *.jpg
│   └── with_helmet/
│       └── *.jpg
├── val/
│   ├── no_helmet/
│   └── with_helmet/
├── test/
│   ├── no_helmet/
│   └── with_helmet/
└── metadata.csv
```

### Thu Thập Dữ Liệu

Dataset được thu thập thông qua ứng dụng `data_collection_app.py`:
- Chụp ảnh trực tiếp từ camera
- Phân loại thủ công: có mũ / không mũ
- Tự động lưu vào thư mục tương ứng

## 🧠 Mô Hình

### Architecture

Model sử dụng **Transfer Learning** với **MobileNetV2** làm base model:

```
Input (224x224x3)
    ↓
MobileNetV2 (Pre-trained on ImageNet)
    ├── Freeze weights (không train)
    └── Output: (7, 7, 1280)
    ↓
Global Average Pooling 2D
    ↓
Dropout (0.5)
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (2 units, Softmax) → Output
```

### Thông Số Model

- **Base Model**: MobileNetV2 (ImageNet weights)
- **Total Params**: 2,422,210
- **Trainable Params**: 164,226
- **Non-trainable Params**: 2,257,984
- **Input Size**: 224x224x3
- **Output Classes**: 2 (no_helmet, with_helmet)

### Hyperparameters

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
OPTIMIZER = Adam
LOSS = Categorical Crossentropy
```

### Callbacks

- **ModelCheckpoint**: Lưu model tốt nhất dựa trên val_accuracy
- **EarlyStopping**: Dừng sớm khi không cải thiện (patience=20)
- **ReduceLROnPlateau**: Giảm learning rate khi loss không cải thiện

## 🔬 Trích Chọn Đặc Trưng

### Transfer Learning với MobileNetV2

Model sử dụng **MobileNetV2** đã được pre-train trên **ImageNet** để trích xuất đặc trưng:

1. **Feature Extraction Layer**: 
   - MobileNetV2 tạo ra feature maps có kích thước (7, 7, 1280)
   - Các đặc trưng này đã được học từ hàng triệu ảnh trong ImageNet

2. **Global Average Pooling**:
   - Chuyển đổi feature maps thành vector 1D có 1280 chiều
   - Giảm tham số và tránh overfitting

3. **Classification Head**:
   - Dense layers để phân loại dựa trên đặc trưng đã trích xuất
   - Dropout layers để regularization

### Data Augmentation

Để tăng cường dữ liệu và cải thiện khả năng generalization:

```python
ImageDataGenerator(
    rescale=1.0/255.0,           # Normalize pixel values
    rotation_range=20,           # Xoay ảnh ±20 độ
    width_shift_range=0.2,       # Dịch chuyển ngang ±20%
    height_shift_range=0.2,      # Dịch chuyển dọc ±20%
    shear_range=0.2,             # Biến dạng shear ±20%
    zoom_range=0.2,              # Zoom ±20%
    horizontal_flip=True,        # Lật ngang ảnh
    fill_mode='nearest'          # Điền pixel gần nhất
)
```

## 📁 Cấu Trúc Dự Án

```
helmet-detection/
│
├── data_collection/              # Thu thập dữ liệu
│   ├── data_collection_app.py   # Ứng dụng chụp ảnh và phân loại
│   ├── no_helmet/               # Ảnh không đội mũ
│   └── with_helmet/             # Ảnh có đội mũ
│
├── dataset/                      # Dataset đã xử lý
│   ├── train/                   # Training set
│   ├── val/                     # Validation set
│   ├── test/                    # Test set
│   └── metadata.csv             # Metadata
│
├── models/                       # Models đã train
│   ├── best_model.h5            # Model tốt nhất
│   ├── final_model.h5           # Model cuối cùng
│   ├── training_history.png     # Đồ thị training
│   └── confusion_matrix.png     # Confusion matrix
│
├── prepare_dataset.py           # Script xử lý dataset
├── train_model.py              # Script training model
├── inference.py                # Script inference/prediction
├── view_dataset.py             # Script xem thông tin dataset
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8+
- TensorFlow 2.10+
- Camera (nếu muốn dùng real-time detection)

### Cài Đặt Dependencies

```bash
# Clone repository
git clone https://github.com/your-username/helmet-detection.git
cd helmet-detection

# Cài đặt dependencies
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
Pillow>=9.0.0
opencv-python>=4.6.0
```

## 📖 Hướng Dẫn Sử Dụng

### 1. Chuẩn Bị Dataset

Nếu bạn chưa có dataset, có thể thu thập dữ liệu bằng ứng dụng:

```bash
python data_collection/data_collection_app.py
```

Sau đó xử lý và tổ chức dataset:

```bash
python prepare_dataset.py
```

Script này sẽ:
- Validate các ảnh
- Chia dataset thành train/val/test (70/15/15)
- Chuẩn hóa tên file
- Tạo metadata CSV

### 2. Xem Thông Tin Dataset

```bash
python view_dataset.py
```

Script này hiển thị:
- Số lượng ảnh trong mỗi split
- Preview một số ảnh mẫu
- Thống kê dataset

### 3. Training Model

```bash
python train_model.py
```

Quá trình training sẽ:
- Tạo data generators với augmentation
- Build model với MobileNetV2
- Train model với callbacks
- Lưu model tốt nhất và final model
- Tạo training history và confusion matrix

### 4. Inference/Prediction

```bash
python inference.py
```

Menu options:
1. **Predict một ảnh**: Nhập đường dẫn ảnh để dự đoán
2. **Predict nhiều ảnh**: Nhập thư mục chứa ảnh để batch processing
3. **Real-time camera**: Nhận dạng real-time qua webcam
4. **Thoát**: Thoát chương trình

## 🎯 Training

### Quy Trình Training

1. **Data Preparation**:
   ```bash
   python prepare_dataset.py
   ```

2. **Training**:
   ```bash
   python train_model.py
   ```

3. **Monitoring**:
   - Training progress được hiển thị real-time
   - Model được lưu tự động khi cải thiện
   - Early stopping sẽ dừng nếu không cải thiện 20 epochs

### Kết Quả Training

- **Best Model**: Lưu tại `models/best_model.h5`
- **Training History**: Lưu tại `models/training_history.png`
- **Confusion Matrix**: Lưu tại `models/confusion_matrix.png`

## 📈 Kết Quả

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **100.00%** |
| **Test Loss** | 0.1990 |
| **Precision** | 1.00 (cả 2 classes) |
| **Recall** | 1.00 (cả 2 classes) |
| **F1-Score** | 1.00 (cả 2 classes) |

### Classification Report

```
              precision    recall  f1-score   support

   no_helmet       1.00      1.00      1.00        13
 with_helmet       1.00      1.00      1.00        12

    accuracy                           1.00        25
   macro avg       1.00      1.00      1.00        25
weighted avg       1.00      1.00      1.00        25
```

### Training Progress

- **Total Epochs Trained**: 37/50 (early stopping)
- **Best Epoch**: 17
- **Best Val Accuracy**: 100%
- **Final Test Accuracy**: 100%

## 🎥 Demo/Inference

### Predict Một Ảnh

```python
# Sử dụng inference.py
python inference.py
# Chọn option 1 và nhập đường dẫn ảnh
```

### Batch Processing

```python
# Xử lý nhiều ảnh trong thư mục
python inference.py
# Chọn option 2 và nhập đường dẫn thư mục
```

### Real-time Camera Detection

```python
# Nhận dạng real-time
python inference.py
# Chọn option 3
# Nhấn 'q' để thoát
```

## 🛠️ Tùy Chỉnh

### Thay Đổi Hyperparameters

Chỉnh sửa trong `train_model.py`:

```python
IMG_SIZE = (224, 224)        # Kích thước ảnh input
BATCH_SIZE = 16              # Batch size
EPOCHS = 50                  # Số epochs
LEARNING_RATE = 0.0001       # Learning rate
```

### Thay Đổi Base Model

Thay thế MobileNetV2 bằng model khác trong `train_model.py`:

```python
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

# Thay MobileNetV2 bằng model khác
base_model = VGG16(...)
```

### Điều Chỉnh Data Augmentation

Chỉnh sửa trong hàm `create_data_generators()`:

```python
train_datagen = ImageDataGenerator(
    rotation_range=30,      # Tăng góc xoay
    zoom_range=0.3,         # Tăng zoom range
    # ...
)
```

## 🔍 Troubleshooting

### Lỗi "Không tìm thấy dataset"

Đảm bảo đã chạy `prepare_dataset.py` trước khi train:

```bash
python prepare_dataset.py
```

### Lỗi "Out of Memory"

Giảm batch size trong `train_model.py`:

```python
BATCH_SIZE = 8  # Giảm từ 16 xuống 8
```

### Model không cải thiện

- Tăng số lượng dữ liệu training
- Điều chỉnh learning rate
- Thử data augmentation mạnh hơn
- Fine-tune base model (unfreeze một số layers)

## 📝 License

Dự án này được phát hành dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 👤 Tác Giả

- **Your Name** - [Your GitHub](https://github.com/your-username)

## 🙏 Acknowledgments

- MobileNetV2 model từ TensorFlow Keras Applications
- Dataset được thu thập và xử lý thủ công
- Sử dụng các thư viện open-source: TensorFlow, Keras, OpenCV, PIL

## 📞 Liên Hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo Issue trên GitHub.

---

**⭐ Nếu dự án hữu ích, hãy cho một star! ⭐**

