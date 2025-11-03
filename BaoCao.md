# BÁO CÁO ĐỒ ÁN

## NHẬN DẠNG MŨ BẢO HIỂM SỬ DỤNG DEEP LEARNING VÀ TRANSFER LEARNING

---

**Tác giả:** [Tên sinh viên]  
**Ngày:** [Ngày tháng năm]  
**Môn học:** [Tên môn học]

---

## MỤC LỤC

1. [MỞ ĐẦU](#1-mở-đầu)
   - 1.1. Giới thiệu vấn đề
   - 1.2. Mục tiêu nghiên cứu
   - 1.3. Phạm vi nghiên cứu
   - 1.4. Cấu trúc báo cáo

2. [CƠ SỞ LÝ THUYẾT](#2-cơ-sở-lý-thuyết)
   - 2.1. Deep Learning và Convolutional Neural Networks (CNN)
   - 2.2. Transfer Learning
   - 2.3. MobileNetV2 Architecture
   - 2.4. Computer Vision và Image Classification

3. [PHƯƠNG PHÁP VÀ CÔNG NGHỆ](#3-phương-pháp-và-công-nghệ)
   - 3.1. Kiến trúc hệ thống
   - 3.2. Xử lý dữ liệu
   - 3.3. Mô hình học máy
   - 3.4. Quy trình huấn luyện
   - 3.5. Đánh giá và Inference

4. [THỰC NGHIỆM](#4-thực-nghiệm)
   - 4.1. Môi trường phát triển
   - 4.2. Thu thập và chuẩn bị dữ liệu
   - 4.3. Huấn luyện mô hình
   - 4.4. Đánh giá kết quả

5. [KẾT QUẢ VÀ ĐÁNH GIÁ](#5-kết-quả-và-đánh-giá)
   - 5.1. Kết quả trên tập Test
   - 5.2. Phân tích hiệu suất
   - 5.3. Visualizations
   - 5.4. So sánh với các phương pháp khác

6. [KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN](#6-kết-luận-và-hướng-phát-triển)
   - 6.1. Kết luận
   - 6.2. Hạn chế
   - 6.3. Hướng phát triển tương lai

7. [TÀI LIỆU THAM KHẢO](#7-tài-liệu-tham-khảo)

8. [PHỤ LỤC](#8-phụ-lục)

---

## 1. MỞ ĐẦU

### 1.1. Giới thiệu vấn đề

An toàn giao thông là một vấn đề quan trọng trong xã hội hiện đại. Việc tuân thủ quy định đội mũ bảo hiểm khi tham gia giao thông là yêu cầu bắt buộc ở nhiều quốc gia, đặc biệt là đối với người điều khiển xe máy. Tuy nhiên, việc kiểm tra và giám sát việc tuân thủ này một cách thủ công là rất khó khăn và tốn kém.

Với sự phát triển của công nghệ Deep Learning và Computer Vision, việc tự động hóa nhận dạng người có đội mũ bảo hiểm hay không trở nên khả thi. Hệ thống này có thể được áp dụng trong:

- Giám sát an toàn giao thông tự động
- Hệ thống cảnh báo tại các khu vực công cộng
- Tích hợp vào hệ thống camera giám sát
- Ứng dụng trong các dự án smart city

### 1.2. Mục tiêu nghiên cứu

Dự án này nhằm mục tiêu:

1. **Xây dựng hệ thống nhận dạng mũ bảo hiểm** sử dụng Deep Learning để phân loại nhị phân: có mũ bảo hiểm / không có mũ bảo hiểm.

2. **Áp dụng Transfer Learning** với MobileNetV2 để tận dụng kiến thức đã được học từ ImageNet, giúp tăng hiệu quả và giảm thời gian huấn luyện.

3. **Đạt độ chính xác cao** trong việc phân loại, đảm bảo tính thực tiễn của hệ thống.

4. **Xây dựng ứng dụng** có thể nhận dạng real-time qua camera hoặc xử lý batch ảnh.

5. **Thiết kế quy trình thu thập và xử lý dữ liệu** hiệu quả, hỗ trợ cả hai phương pháp: chuẩn hóa đơn giản và xử lý nâng cao với OpenCV.

### 1.3. Phạm vi nghiên cứu

Dự án tập trung vào:

- **Phân loại nhị phân**: Chỉ phân biệt 2 lớp - có mũ bảo hiểm và không có mũ bảo hiểm
- **Input**: Ảnh tĩnh hoặc video stream từ camera
- **Output**: Nhãn phân loại kèm confidence score
- **Dataset**: Thu thập thủ công qua ứng dụng camera, tổng cộng 149 ảnh
- **Model**: Sử dụng MobileNetV2 làm base model với Transfer Learning

**Giới hạn:**
- Chưa hỗ trợ detection nhiều người trong một ảnh
- Chưa nhận dạng các loại mũ bảo hiểm khác nhau
- Chưa xử lý các trường hợp đặc biệt (mũ che mặt, góc nhìn nghiêng...)

### 1.4. Cấu trúc báo cáo

Báo cáo được tổ chức thành các chương chính:
- **Chương 2**: Trình bày cơ sở lý thuyết về Deep Learning, CNN, Transfer Learning
- **Chương 3**: Mô tả chi tiết phương pháp và công nghệ được sử dụng
- **Chương 4**: Mô tả quá trình thực nghiệm, thu thập dữ liệu và huấn luyện
- **Chương 5**: Phân tích kết quả và đánh giá hiệu suất
- **Chương 6**: Kết luận và đề xuất hướng phát triển

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Deep Learning và Convolutional Neural Networks (CNN)

#### 2.1.1. Deep Learning

Deep Learning là một nhánh của Machine Learning sử dụng các mạng neural với nhiều lớp (deep networks) để học các đặc trưng (features) từ dữ liệu. Khác với Machine Learning truyền thống, Deep Learning có khả năng tự động học các đặc trưng phức tạp từ dữ liệu thô mà không cần thiết kế thủ công.

**Ưu điểm của Deep Learning:**
- Tự động học đặc trưng từ dữ liệu
- Có khả năng xử lý dữ liệu lớn và phức tạp
- Hiệu suất cao trong các bài toán Computer Vision
- Có thể transfer knowledge từ task này sang task khác

#### 2.1.2. Convolutional Neural Networks (CNN)

CNN là kiến trúc mạng neural được thiết kế đặc biệt cho xử lý dữ liệu có cấu trúc grid như hình ảnh. CNN bao gồm các thành phần chính:

**a) Convolutional Layer (Lớp Tích chập):**
- Áp dụng các bộ lọc (filters/kernels) để phát hiện các đặc trưng cục bộ
- Giảm số lượng tham số so với Fully Connected Layer
- Bảo toàn mối quan hệ không gian trong ảnh

**b) Pooling Layer (Lớp Gộp):**
- Giảm kích thước không gian của feature maps
- Giảm overfitting và tính toán
- Các loại phổ biến: Max Pooling, Average Pooling

**c) Fully Connected Layer (Lớp Kết nối đầy đủ):**
- Kết nối tất cả neurons từ lớp trước
- Thực hiện phân loại cuối cùng

**d) Activation Functions:**
- ReLU (Rectified Linear Unit): f(x) = max(0, x)
- Softmax: Chuyển đổi logits thành xác suất

### 2.2. Transfer Learning

#### 2.2.1. Khái niệm Transfer Learning

Transfer Learning là kỹ thuật tận dụng kiến thức đã được học từ một task/dataset để áp dụng vào một task/dataset mới. Thay vì train từ đầu, ta sử dụng một pre-trained model đã được huấn luyện trên dataset lớn (như ImageNet) làm điểm khởi đầu.

**Lợi ích của Transfer Learning:**
- Tiết kiệm thời gian và tài nguyên tính toán
- Đạt hiệu suất cao ngay cả với dataset nhỏ
- Tận dụng kiến thức đã được học từ hàng triệu ảnh
- Giảm overfitting

#### 2.2.2. Các phương pháp Transfer Learning

**a) Feature Extraction:**
- Giữ nguyên pre-trained model, freeze các layers
- Chỉ train các lớp phân loại (classification head) mới
- Sử dụng khi dataset nhỏ và tương tự dataset gốc

**b) Fine-tuning:**
- Unfreeze một phần hoặc toàn bộ pre-trained model
- Train lại với learning rate nhỏ
- Sử dụng khi có dataset lớn hơn hoặc khác biệt

**Trong dự án này:** Sử dụng Feature Extraction với MobileNetV2 được freeze hoàn toàn.

### 2.3. MobileNetV2 Architecture

#### 2.3.1. Giới thiệu MobileNetV2

MobileNetV2 là một kiến trúc CNN được Google phát triển năm 2018, được tối ưu hóa cho các thiết bị di động và embedded systems. Đây là phiên bản cải tiến của MobileNetV1.

**Đặc điểm chính:**
- **Depthwise Separable Convolution**: Giảm số lượng tham số và tính toán
- **Inverted Residuals**: Cải thiện gradient flow và hiệu suất
- **Linear Bottlenecks**: Thay thế ReLU6 ở bottleneck để giữ thông tin
- **Nhẹ và nhanh**: Chỉ ~3.5M parameters, phù hợp cho mobile/edge devices

#### 2.3.2. Depthwise Separable Convolution

Thay vì một convolution thông thường, Depthwise Separable Convolution chia thành 2 bước:

**Bước 1: Depthwise Convolution**
- Mỗi input channel được filter riêng biệt
- Không kết hợp thông tin giữa các channels

**Bước 2: Pointwise Convolution (1x1 Convolution)**
- Kết hợp thông tin từ các channels
- Sử dụng filter 1x1

**Lợi ích:**
- Giảm số lượng tham số từ D_K × D_K × M × N xuống D_K × D_K × M + M × N
- Giảm tính toán và memory footprint

#### 2.3.3. Inverted Residuals với Linear Bottlenecks

**Inverted Residuals:**
- Expand: Mở rộng số channels (thường x6)
- Depthwise: Filter không gian
- Project: Thu hẹp lại số channels

**Linear Bottlenecks:**
- Sử dụng linear activation thay vì ReLU ở bottleneck
- Giữ lại thông tin quan trọng không bị mất do ReLU

#### 2.3.4. Tại sao chọn MobileNetV2?

1. **Hiệu suất cao**: Đạt độ chính xác tốt trên ImageNet với ít tham số
2. **Nhẹ**: Chỉ ~3.5M parameters, phù hợp cho real-time inference
3. **Nhanh**: Tối ưu cho inference trên CPU và mobile devices
4. **Pre-trained tốt**: Có sẵn weights được train trên ImageNet
5. **Cân bằng**: Tốt giữa accuracy và efficiency

### 2.4. Computer Vision và Image Classification

#### 2.4.1. Image Classification

Image Classification là bài toán cơ bản trong Computer Vision, mục tiêu là gán một nhãn cho một ảnh đầu vào từ một tập các nhãn đã định nghĩa trước.

**Quy trình:**
1. **Preprocessing**: Chuẩn hóa ảnh (resize, normalize)
2. **Feature Extraction**: Trích xuất đặc trưng (bằng CNN)
3. **Classification**: Phân loại dựa trên đặc trưng
4. **Post-processing**: Xử lý kết quả (softmax, threshold)

#### 2.4.2. Data Augmentation

Data Augmentation là kỹ thuật tăng cường dữ liệu bằng cách tạo ra các biến thể của ảnh gốc, giúp:
- Tăng kích thước dataset
- Giảm overfitting
- Cải thiện khả năng generalization

**Các phép biến đổi phổ biến:**
- Rotation: Xoay ảnh
- Translation: Dịch chuyển ảnh
- Scaling: Thay đổi kích thước
- Flipping: Lật ảnh
- Color jittering: Thay đổi màu sắc, độ sáng
- Cropping: Cắt ảnh

---

## 3. PHƯƠNG PHÁP VÀ CÔNG NGHỆ

### 3.1. Kiến trúc hệ thống

Hệ thống được chia thành các module chính:

```
┌─────────────────────────────────────────────────┐
│         HELMET DETECTION SYSTEM                 │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. DATA COLLECTION MODULE                       │
│     - data_collection_app.py                    │
│     - Thu thập ảnh qua camera                    │
│     - Phân loại thủ công                        │
│                                                  │
│  2. DATA PREPROCESSING MODULE                    │
│     - prepare_dataset.py (Cách 1: Đơn giản)     │
│     - prepare_dataset_2.py (Cách 2: OpenCV)    │
│     - Xử lý và chuẩn bị dataset                 │
│                                                  │
│  3. TRAINING MODULE                             │
│     - train_model.py                            │
│     - Xây dựng và huấn luyện model              │
│                                                  │
│  4. INFERENCE MODULE                            │
│     - inference.py                              │
│     - Prediction từ ảnh/camera                  │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 3.2. Xử lý dữ liệu

#### 3.2.1. Thu thập dữ liệu

**Ứng dụng thu thập dữ liệu:**
- File: `data_collection/data_collection_app.py`
- Giao diện GUI sử dụng Tkinter
- Chức năng:
  - Kết nối camera và hiển thị video stream
  - Phát hiện khuôn mặt tự động (Haar Cascade)
  - Chụp ảnh và lưu vào thư mục tương ứng
  - Phân loại thủ công: "with_helmet" hoặc "no_helmet"

**Cấu trúc dữ liệu gốc:**
```
data_collection/
├── with_helmet/
│   └── helmet_*.jpg
└── no_helmet/
    └── no_helmet_*.jpg
```

#### 3.2.2. Tiền xử lý dữ liệu

Dự án hỗ trợ **2 phương pháp** xử lý dataset:

**Cách 1: Chuẩn hóa đơn giản** (`prepare_dataset.py`)
- Validate ảnh (kiểm tra corrupt)
- Chia dataset: Train (70%) / Val (15%) / Test (15%)
- Copy và đổi tên file theo chuẩn: `class_split_index.jpg`
- Giữ nguyên kích thước ảnh gốc
- Tạo metadata CSV

**Cách 2: Xử lý với OpenCV** (`prepare_dataset_2.py`)
- **Face Detection**: Sử dụng Haar Cascade để phát hiện khuôn mặt
- **Crop và mở rộng**: Crop vùng face + mở rộng 30% phía trên để bao gồm mũ
- **Image Enhancement**: Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) để cải thiện contrast
- **Resize**: Chuẩn hóa tất cả ảnh về 224x224
- **Quality Check**: Lọc ảnh có kích thước tối thiểu 100x100
- Chia dataset và tạo metadata

**So sánh 2 cách:**

| Tiêu chí | Cách 1 | Cách 2 |
|----------|--------|--------|
| Tốc độ | Nhanh | Chậm hơn |
| Kích thước ảnh | Khác nhau | Đồng nhất 224x224 |
| Face Detection | Không | Có |
| Image Enhancement | Không | CLAHE |
| Tập trung vùng quan trọng | Toàn ảnh | Face + mũ |
| Số lượng ảnh | Tất cả (149) | Có thể ít hơn (~123) |

**Cấu trúc dataset sau xử lý:**
```
dataset/
├── train/
│   ├── with_helmet/
│   │   └── with_helmet_train_0000.jpg
│   └── no_helmet/
│       └── no_helmet_train_0000.jpg
├── val/
│   ├── with_helmet/
│   └── no_helmet/
├── test/
│   ├── with_helmet/
│   └── no_helmet/
└── metadata.csv
```

#### 3.2.3. Data Augmentation

Trong quá trình training, sử dụng `ImageDataGenerator` của Keras để tăng cường dữ liệu:

```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,        # Normalize pixel values [0,1]
    rotation_range=20,         # Xoay ±20 độ
    width_shift_range=0.2,    # Dịch ngang ±20%
    height_shift_range=0.2,   # Dịch dọc ±20%
    shear_range=0.2,          # Biến dạng shear ±20%
    zoom_range=0.2,           # Zoom ±20%
    horizontal_flip=True,     # Lật ngang
    fill_mode='nearest'       # Điền pixel gần nhất
)
```

**Lưu ý:** 
- Chỉ áp dụng augmentation cho tập training
- Validation và Test chỉ normalize (rescale), không augmentation

### 3.3. Mô hình học máy

#### 3.3.1. Kiến trúc Model

**Architecture:**

```
Input: (224, 224, 3)
    ↓
MobileNetV2 (Pre-trained on ImageNet)
    ├── include_top=False (không có classification head)
    ├── weights='imagenet' (sử dụng pre-trained weights)
    ├── Freeze: base_model.trainable = False
    └── Output: (7, 7, 1280) feature maps
    ↓
GlobalAveragePooling2D()
    └── Output: (1280,) vector
    ↓
Dropout(0.5)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(2, activation='softmax')
    └── Output: [no_helmet_prob, with_helmet_prob]
```

**Code implementation:**

```python
# Load MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Build model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])
```

#### 3.3.2. Thông số Model

- **Base Model**: MobileNetV2
- **Total Parameters**: 2,422,210
- **Trainable Parameters**: 164,226 (chỉ các lớp Dense và Dropout)
- **Non-trainable Parameters**: 2,257,984 (MobileNetV2 được freeze)
- **Input Size**: 224×224×3 (RGB)
- **Output Classes**: 2 (no_helmet, with_helmet)

#### 3.3.3. Loss Function và Optimizer

- **Loss Function**: `categorical_crossentropy`
  - Phù hợp cho bài toán phân loại đa lớp với one-hot encoding
  
- **Optimizer**: `Adam` với learning rate = 0.0001
  - Adam là adaptive learning rate optimizer
  - Learning rate nhỏ phù hợp cho Transfer Learning
  
- **Metrics**: `accuracy`
  - Theo dõi độ chính xác trong quá trình training

#### 3.3.4. Callbacks

Để tối ưu quá trình training, sử dụng 3 callbacks:

**a) ModelCheckpoint:**
```python
ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```
- Lưu model có validation accuracy cao nhất

**b) EarlyStopping:**
```python
EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)
```
- Dừng sớm khi validation accuracy không cải thiện trong 20 epochs
- Khôi phục weights tốt nhất

**c) ReduceLROnPlateau:**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)
```
- Giảm learning rate khi validation loss không cải thiện
- Factor 0.5: giảm còn một nửa
- Min learning rate: 0.00001

### 3.4. Quy trình huấn luyện

#### 3.4.1. Hyperparameters

```python
IMG_SIZE = (224, 224)      # Kích thước input
BATCH_SIZE = 16            # Số ảnh mỗi batch
EPOCHS = 50                # Số epochs tối đa
LEARNING_RATE = 0.0001     # Learning rate
NUM_CLASSES = 2            # Số lớp phân loại
```

#### 3.4.2. Quy trình Training

1. **Kiểm tra dataset**: Đảm bảo folder `dataset` tồn tại

2. **Tạo Data Generators**: 
   - Train generator với augmentation
   - Val generator chỉ normalize
   - Test generator chỉ normalize

3. **Xây dựng Model**:
   - Load MobileNetV2 pre-trained
   - Freeze base model
   - Thêm classification head
   - Compile với optimizer, loss, metrics

4. **Training**:
   - Train với train_generator
   - Validate với val_generator
   - Sử dụng callbacks để tối ưu

5. **Đánh giá**:
   - Evaluate trên test set
   - Tính confusion matrix
   - Tính classification report

6. **Lưu kết quả**:
   - Lưu best model
   - Vẽ training history
   - Vẽ confusion matrix

### 3.5. Đánh giá và Inference

#### 3.5.1. Metrics đánh giá

**Accuracy (Độ chính xác):**
- Tỷ lệ dự đoán đúng trên tổng số mẫu
- Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Precision (Độ chính xác dự đoán):**
- Precision = TP / (TP + FP)
- Đo lường khả năng dự đoán đúng trong số các dự đoán dương tính

**Recall (Độ nhạy):**
- Recall = TP / (TP + FN)
- Đo lường khả năng phát hiện các mẫu dương tính thực sự

**F1-Score:**
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Cân bằng giữa Precision và Recall

**Confusion Matrix:**
- Ma trận biểu thị số lượng dự đoán đúng/sai cho mỗi lớp

#### 3.5.2. Inference Pipeline

**Preprocessing:**
1. Load ảnh và resize về 224×224
2. Chuyển đổi sang array
3. Normalize: pixel values / 255.0
4. Expand dimensions: (1, 224, 224, 3)

**Prediction:**
1. Model.predict() → probabilities [p_no_helmet, p_with_helmet]
2. Argmax để lấy class có probability cao nhất
3. Confidence = max probability × 100%

**Output:**
- Predicted class: "no_helmet" hoặc "with_helmet"
- Confidence score (%)
- Probabilities cho cả 2 classes

#### 3.5.3. Các chế độ Inference

**a) Predict một ảnh:**
- Input: Đường dẫn file ảnh
- Output: Class + Confidence + Visualization

**b) Batch Processing:**
- Input: Thư mục chứa nhiều ảnh
- Output: Danh sách kết quả cho tất cả ảnh

**c) Real-time Camera:**
- Input: Video stream từ webcam
- Output: Overlay kết quả lên frame
- Real-time prediction và hiển thị

---

## 4. THỰC NGHIỆM

### 4.1. Môi trường phát triển

#### 4.1.1. Phần cứng

- **CPU**: Tùy theo máy sử dụng
- **RAM**: Khuyến nghị ≥ 8GB
- **GPU**: Không bắt buộc (có GPU sẽ train nhanh hơn)
- **Camera**: Webcam để test real-time (nếu có)

#### 4.1.2. Phần mềm

- **Hệ điều hành**: Windows 10/11, Linux, macOS
- **Python**: Version 3.8 trở lên
- **Thư viện chính**:
  - TensorFlow >= 2.10.0
  - Keras (tích hợp trong TensorFlow)
  - NumPy >= 1.21.0
  - OpenCV >= 4.6.0
  - Pandas >= 1.3.0
  - Matplotlib >= 3.5.0
  - Seaborn >= 0.12.0
  - scikit-learn >= 1.0.0
  - Pillow >= 9.0.0

**Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

### 4.2. Thu thập và chuẩn bị dữ liệu

#### 4.2.1. Thu thập dữ liệu

**Bước 1: Chạy ứng dụng thu thập dữ liệu**
```bash
python data_collection/data_collection_app.py
```

**Bước 2: Sử dụng ứng dụng**
1. Nhấn "Start Camera" để bật webcam
2. Đặt người trong khung hình (có hoặc không có mũ bảo hiểm)
3. Nhấn "Capture - With Helmet" hoặc "Capture - No Helmet" để chụp
4. Ảnh tự động lưu vào thư mục tương ứng
5. Lặp lại cho đến khi có đủ số lượng ảnh

**Kết quả:**
- Tổng số ảnh: 149 ảnh
- `with_helmet`: 72 ảnh
- `no_helmet`: 79 ảnh

#### 4.2.2. Chuẩn bị Dataset

**Phương pháp 1: Chuẩn hóa đơn giản**

```bash
python scripts/data_preprocessing/prepare_dataset.py
```

**Quy trình:**
1. Validate tất cả ảnh (kiểm tra corrupt)
2. Chia dataset: 70% train, 15% val, 15% test
3. Copy và đổi tên file theo chuẩn
4. Tạo metadata CSV

**Kết quả:**
- Train: 103 ảnh (no_helmet: 54, with_helmet: 49)
- Validation: 21 ảnh (no_helmet: 11, with_helmet: 10)
- Test: 25 ảnh (no_helmet: 13, with_helmet: 12)

**Phương pháp 2: Xử lý với OpenCV**

```bash
python scripts/data_preprocessing/prepare_dataset_2.py
```

**Quy trình:**
1. Load Haar Cascade cho face detection
2. Với mỗi ảnh:
   - Detect face
   - Crop và mở rộng vùng face (thêm 30% phía trên)
   - Áp dụng CLAHE để enhance
   - Resize về 224×224
3. Quality check và lọc
4. Chia dataset và tạo metadata

**Kết quả:** 
- Số ảnh có thể ít hơn do lọc quality và không detect được face
- Tất cả ảnh được chuẩn hóa về 224×224

**Sử dụng menu (Khuyến nghị):**
```bash
python scripts/data_preprocessing/prepare_dataset_main.py
```

Menu cho phép chọn giữa 2 phương pháp.

### 4.3. Huấn luyện mô hình

#### 4.3.1. Chạy Training

```bash
python scripts/training/train_model.py
```

#### 4.3.2. Quá trình Training

**Bước 1: Tạo Data Generators**
- Train generator với augmentation
- Val generator chỉ normalize
- Test generator chỉ normalize

**Bước 2: Xây dựng Model**
- Load MobileNetV2 với ImageNet weights
- Freeze base model
- Thêm classification head
- Compile với Adam optimizer

**Bước 3: Training**
- Fit model với train và validation data
- Sử dụng callbacks để tối ưu
- Model được lưu tự động khi cải thiện

**Bước 4: Đánh giá**
- Evaluate trên test set
- Tính các metrics
- Vẽ confusion matrix và training history

#### 4.3.3. Kết quả Training

**Training Progress:**
- Total epochs trained: 37/50 (early stopping)
- Best epoch: 17
- Best validation accuracy: 100%
- Training dừng sớm do EarlyStopping (không cải thiện trong 20 epochs)

**Files được tạo:**
- `models/best_model.h5`: Model tốt nhất
- `models/training_history.png`: Đồ thị training history
- `models/confusion_matrix.png`: Confusion matrix

### 4.4. Đánh giá kết quả

Xem chi tiết trong **Chương 5: Kết quả và Đánh giá**.

---

## 5. KẾT QUẢ VÀ ĐÁNH GIÁ

### 5.1. Kết quả trên tập Test

#### 5.1.1. Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **100.00%** |
| **Test Loss** | 0.1990 |
| **Precision** | 1.00 (cả 2 classes) |
| **Recall** | 1.00 (cả 2 classes) |
| **F1-Score** | 1.00 (cả 2 classes) |

#### 5.1.2. Classification Report

```
              precision    recall  f1-score   support

   no_helmet       1.00      1.00      1.00        13
 with_helmet       1.00      1.00      1.00        12

    accuracy                           1.00        25
   macro avg       1.00      1.00      1.00        25
weighted avg       1.00      1.00      1.00        25
```

**Phân tích:**
- Model đạt **100% accuracy** trên test set
- **Perfect Precision và Recall** cho cả 2 classes
- Không có False Positive hay False Negative
- 25 mẫu test: 13 no_helmet, 12 with_helmet

#### 5.1.3. Confusion Matrix

```
                Predicted
Actual      no_helmet  with_helmet
no_helmet      13          0
with_helmet     0         12
```

**Nhận xét:**
- **True Positive (TP)**: 25 (tất cả dự đoán đúng)
- **False Positive (FP)**: 0
- **False Negative (FN)**: 0
- **True Negative (TN)**: N/A (bài toán nhị phân)

### 5.2. Phân tích hiệu suất

#### 5.2.1. Training History

**Accuracy Curve:**
- Train accuracy: Tăng nhanh và ổn định
- Val accuracy: Đạt 100% tại epoch 17
- Không có dấu hiệu overfitting

**Loss Curve:**
- Train loss: Giảm nhanh và ổn định
- Val loss: Giảm và hội tụ
- Không có dấu hiệu overfitting

**Nhận xét:**
- Model học tốt, không bị overfitting
- Early stopping hoạt động hiệu quả
- Convergence nhanh nhờ Transfer Learning

#### 5.2.2. Ưu điểm của Model

1. **Độ chính xác cao**: 100% trên test set
2. **Nhẹ và nhanh**: MobileNetV2 chỉ ~3.5M parameters
3. **Real-time**: Có thể inference real-time trên CPU
4. **Transfer Learning hiệu quả**: Tận dụng kiến thức từ ImageNet
5. **Không overfitting**: Training và validation metrics khớp nhau

#### 5.2.3. Hạn chế

1. **Dataset nhỏ**: Chỉ 149 ảnh, có thể không đại diện đầy đủ
2. **Test set nhỏ**: 25 mẫu test có thể không đủ để đánh giá đầy đủ
3. **Chưa kiểm tra trên dữ liệu ngoài**: Chưa test trên ảnh từ nguồn khác
4. **Chưa xử lý edge cases**: Góc nhìn nghiêng, ánh sáng yếu, v.v.

### 5.3. Visualizations

#### 5.3.1. Training History Plot

File: `models/training_history.png`

- **Subplot 1**: Accuracy (Train vs Validation)
- **Subplot 2**: Loss (Train vs Validation)

#### 5.3.2. Confusion Matrix Plot

File: `models/confusion_matrix.png`

- Heatmap hiển thị confusion matrix
- Các giá trị được annotate rõ ràng

### 5.4. So sánh với các phương pháp khác

#### 5.4.1. So sánh với các Model khác

| Model | Parameters | Accuracy | Inference Time | Use Case |
|-------|-----------|----------|----------------|----------|
| **MobileNetV2** (dự án) | ~3.5M | 100% | Fast | Mobile/Edge |
| VGG16 | ~138M | High | Slow | Server |
| ResNet50 | ~25M | Very High | Medium | Server |
| EfficientNet-B0 | ~11M | Very High | Medium | Flexible |

**Kết luận:**
- MobileNetV2 là lựa chọn tốt cho bài toán này
- Cân bằng giữa accuracy và efficiency
- Phù hợp cho real-time applications

#### 5.4.2. So sánh với Training từ đầu

| Phương pháp | Thời gian training | Accuracy | Dataset yêu cầu |
|-------------|-------------------|----------|-----------------|
| Transfer Learning | Ngắn (~37 epochs) | 100% | Nhỏ (149 ảnh) |
| Training từ đầu | Dài (nhiều epochs) | Không chắc chắn | Lớn (hàng nghìn) |

**Kết luận:**
- Transfer Learning hiệu quả hơn rất nhiều
- Tiết kiệm thời gian và tài nguyên
- Đạt kết quả tốt ngay cả với dataset nhỏ

---

## 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận

Dự án đã thành công trong việc xây dựng hệ thống nhận dạng mũ bảo hiểm sử dụng Deep Learning và Transfer Learning. Các kết quả đạt được:

1. **Xây dựng thành công hệ thống** với kiến trúc MobileNetV2 + Transfer Learning
2. **Đạt độ chính xác 100%** trên test set
3. **Xây dựng pipeline hoàn chỉnh**: Từ thu thập dữ liệu đến inference
4. **Hỗ trợ nhiều chế độ**: Single image, batch processing, real-time camera
5. **Code được tổ chức tốt**: Dễ maintain và mở rộng

**Đóng góp:**
- Áp dụng thành công Transfer Learning cho bài toán nhận dạng mũ bảo hiểm
- Xây dựng quy trình thu thập và xử lý dữ liệu hiệu quả
- Tạo ứng dụng có thể sử dụng thực tế

### 6.2. Hạn chế

1. **Dataset nhỏ**: 149 ảnh có thể không đủ đại diện
2. **Test set nhỏ**: 25 mẫu có thể không đủ để đánh giá đầy đủ
3. **Chưa kiểm tra generalization**: Chưa test trên dữ liệu từ nguồn khác
4. **Chưa xử lý edge cases**: Góc nhìn nghiêng, điều kiện ánh sáng khác nhau
5. **Chưa hỗ trợ multi-person**: Chỉ phân loại 1 người trong ảnh
6. **Chưa phân loại loại mũ**: Chỉ phân biệt có/không có mũ

### 6.3. Hướng phát triển tương lai

#### 6.3.1. Cải thiện Dataset

1. **Mở rộng dataset**:
   - Thu thập thêm nhiều ảnh đa dạng
   - Thêm ảnh từ các góc độ, điều kiện ánh sáng khác nhau
   - Thêm ảnh từ nhiều loại mũ bảo hiểm khác nhau

2. **Data augmentation nâng cao**:
   - Mixup, CutMix
   - Advanced color transformations
   - Style transfer

#### 6.3.2. Cải thiện Model

1. **Fine-tuning**:
   - Unfreeze một số layers cuối của MobileNetV2
   - Fine-tune với learning rate nhỏ

2. **Thử các model khác**:
   - EfficientNet series
   - Vision Transformer (ViT)
   - Ensemble models

3. **Optimization**:
   - Model quantization
   - Model pruning
   - TensorFlow Lite cho mobile deployment

#### 6.3.3. Tính năng mới

1. **Multi-person detection**:
   - Kết hợp với object detection (YOLO, SSD)
   - Phát hiện nhiều người trong một ảnh

2. **Classification nâng cao**:
   - Phân loại loại mũ bảo hiểm
   - Phát hiện mũ bị lệch, không đúng cách

3. **Real-time video processing**:
   - Xử lý video stream
   - Tracking người qua các frame
   - Cảnh báo khi phát hiện không đội mũ

4. **Deployment**:
   - Web application (Flask/FastAPI)
   - Mobile app (Android/iOS)
   - Edge devices (Raspberry Pi, Jetson Nano)

#### 6.3.4. Hệ thống hoàn chỉnh

1. **Backend API**:
   - RESTful API để nhận ảnh và trả kết quả
   - Database để lưu trữ kết quả
   - User authentication

2. **Frontend Dashboard**:
   - Web interface để upload ảnh
   - Hiển thị statistics
   - Quản lý dataset

3. **Integration**:
   - Tích hợp vào hệ thống camera giám sát
   - Kết nối với hệ thống báo cảnh
   - Export reports

---

## 7. TÀI LIỆU THAM KHẢO

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

2. Howard, A., et al. (2017). **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**. arXiv preprint arXiv:1704.04861.

3. Tan, M., & Le, Q. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**. International Conference on Machine Learning.

4. Keras Documentation. **Transfer Learning Guide**. https://keras.io/guides/transfer_learning/

5. TensorFlow Documentation. **MobileNetV2**. https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2

6. Chollet, F. (2017). **Deep Learning with Python**. Manning Publications.

7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). **Deep Learning**. MIT Press.

8. OpenCV Documentation. **Haar Cascades**. https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html

9. Scikit-learn Documentation. **Classification Metrics**. https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

10. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition**. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

---

## 8. PHỤ LỤC

### Phụ lục: Cấu trúc Project

```
helmet-detection/
├── data_collection/              # Thu thập dữ liệu
│   ├── data_collection_app.py   # Ứng dụng chụp ảnh
│   ├── with_helmet/             # Ảnh có mũ
│   └── no_helmet/               # Ảnh không mũ
│
├── dataset/                      # Dataset đã xử lý
│   ├── train/                    # Training set
│   ├── val/                      # Validation set
│   ├── test/                     # Test set
│   └── *.csv                     # Metadata
│
├── models/                       # Models đã train
│   ├── best_model.h5            # Model tốt nhất
│   ├── training_history.png     # Training curves
│   └── confusion_matrix.png     # Confusion matrix
│
├── scripts/                      # Scripts
│   ├── data_preprocessing/      # Xử lý dữ liệu
│   ├── training/                # Training
│   ├── inference/               # Inference
│   └── utils/                   # Utilities
│
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── BaoCao.doc                   # Báo cáo này
```