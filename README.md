# 🚗 Hệ Thống Nhận Diện Biển Số Xe (ALPR)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.x-00B4D8?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

> **Hệ thống nhận diện biển số xe tự động (ALPR)** kết hợp xử lý ảnh đa tầng,  
> phát hiện đối tượng YOLOv8 và nhận dạng ký tự EasyOCR cho biển số xe Việt Nam.

<br/>

[📖 Tổng quan](#-tổng-quan) •
[⚙️ Cài đặt](#️-cài-đặt) •
[🚀 Sử dụng](#-sử-dụng) •
[🏗️ Kiến trúc](#️-kiến-trúc-hệ-thống) •
[📊 Kết quả](#-kết-quả-thực-nghiệm) •
[👨‍💻 Nhóm](#-nhóm-thực-hiện)

</div>

---

## 📖 Tổng quan

Dự án xây dựng hệ thống **Automatic License Plate Recognition (ALPR)** dành cho biển số xe Việt Nam, được phát triển bởi nhóm sinh viên Khoa Công nghệ Thông tin – Đại học Xây dựng Hà Nội.

Hệ thống giải quyết bài toán nhận diện biển số theo ba giai đoạn:

```
Ảnh đầu vào  →  Tiền xử lý ảnh  →  Phát hiện biển số  →  Nhận dạng ký tự  →  Kết quả
```

### ✨ Điểm nổi bật

| Tính năng | Mô tả |
|---|---|
| 🔍 **Tiền xử lý đa tầng** | HSV → CLAHE → Bilateral Filter → Sharpening |
| 🎯 **Phát hiện chính xác** | YOLOv8n với mAP > 97%, tốc độ > 30 FPS (GPU) |
| 📝 **Nhận dạng ký tự** | EasyOCR hỗ trợ biển số 1 và 2 dòng |
| 🖥️ **CPU-friendly** | Thời gian xử lý 137ms/ảnh, không cần GPU |
| 📐 **Hỗ trợ đa dạng** | Biển số dài (470×110mm) và ngắn (280×200mm) |

---

## 📁 Cấu trúc dự án

```
license-plate-recognition/
│
├── 📂 data/
│   ├── images/                  # Ảnh đầu vào
│   ├── labels/                  # Nhãn YOLO format
│   └── license_plate.yaml       # Config dataset
│
├── 📂 models/
│   ├── yolov8n.pt               # Pretrained YOLOv8 Nano
│   └── best_license_plate_detector.pt  # Model đã finetune
│
├── 📂 src/
│   ├── preprocessing.py         # Các bước tiền xử lý ảnh
│   ├── detection.py             # Phát hiện biển số (YOLO)
│   ├── segmentation.py          # Phân đoạn ký tự
│   ├── recognition.py           # Nhận dạng ký tự (EasyOCR)
│   └── pipeline.py              # Pipeline tổng hợp end-to-end
│
├── 📂 utils/
│   ├── evaluate.py              # Đánh giá hiệu năng
│   └── visualize.py             # Hiển thị kết quả
│
├── 📂 notebooks/
│   └── demo.ipynb               # Notebook demo
│
├── 📂 results/
│   └── output_images/           # Ảnh kết quả
│
├── train.py                     # Script huấn luyện YOLOv8
├── inference.py                 # Script chạy nhận diện
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống

- Python **3.8** trở lên
- RAM tối thiểu **4GB** (khuyến nghị 8GB)
- GPU tùy chọn (CUDA 11.8+ nếu có)

### 1. Clone repository

```bash
git clone https://github.com/<username>/license-plate-recognition.git
cd license-plate-recognition
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
easyocr>=1.7.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
matplotlib>=3.7.0
PyYAML>=6.0
```

---

## 🚀 Sử dụng

### Nhận diện ảnh đơn lẻ

```python
from src.pipeline import LicensePlateRecognizer

# Khởi tạo hệ thống
recognizer = LicensePlateRecognizer(model_path='models/best_license_plate_detector.pt')

# Nhận diện biển số
result = recognizer.recognize('data/images/car.jpg')
print(f"Biển số: {result['plate_text']}")
print(f"Độ tin cậy: {result['confidence']:.2f}")
```

### Chạy từ command line

```bash
# Nhận diện một ảnh
python inference.py --image path/to/image.jpg

# Nhận diện từ webcam
python inference.py --source 0

# Nhận diện video
python inference.py --source path/to/video.mp4

# Chỉ định model tùy chỉnh
python inference.py --image car.jpg --model models/best_license_plate_detector.pt
```

### Huấn luyện lại mô hình

```bash
python train.py --data data/license_plate.yaml --epochs 100 --batch 16
```

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        ẢNH ĐẦU VÀO                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │        TIỀN XỬ LÝ ẢNH        │
          │  ┌─────────────────────────┐  │
          │  │  1. HSV – Kênh Value    │  │
          │  │  2. Resize chuẩn hóa   │  │
          │  │  3. CLAHE tương phản   │  │
          │  │  4. Bilateral Filter   │  │
          │  │  5. Sharpening kernel  │  │
          │  │  6. Adaptive Threshold │  │
          │  │  7. Morphology Closing │  │
          │  └─────────────────────────┘  │
          └───────────────┬───────────────┘
                          │
          ┌───────────────▼───────────────┐
          │    PHÁT HIỆN BIỂN SỐ (YOLO)   │
          │       YOLOv8n – mAP > 97%     │
          └───────────────┬───────────────┘
                          │
          ┌───────────────▼───────────────┐
          │      PHÂN ĐOẠN KÝ TỰ          │
          │  Dilation → Contour → IoU     │
          │  Sắp xếp trái→phải, trên→dưới│
          └───────────────┬───────────────┘
                          │
          ┌───────────────▼───────────────┐
          │    NHẬN DẠNG KÝ TỰ (OCR)      │
          │     EasyOCR – Confidence>0.5  │
          └───────────────┬───────────────┘
                          │
          ┌───────────────▼───────────────┐
          │          KẾT QUẢ              │
          │    Chuỗi biển số + Score      │
          └───────────────────────────────┘
```

---

## 🔬 Chi tiết các bước xử lý

### Bước 1 – Chuyển sang ảnh xám (HSV)

```python
def convert_to_grayscale(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2]  # Kênh Value
    return gray
```

> Sử dụng kênh **Value** của HSV giúp loại bỏ nhiễu màu sắc môi trường, tốt hơn so với grayscale thông thường.

---

### Bước 2 – Tăng cường tương phản (CLAHE)

```python
def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)
```

> **CLAHE** (Contrast Limited Adaptive Histogram Equalization) xử lý ánh sáng không đồng đều – tăng tương phản **1.28x** so với baseline.

---

### Bước 3 – Khử nhiễu Bilateral

```python
denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
```

> Bảo toàn biên ký tự trong khi loại bỏ nhiễu hạt – điểm khác biệt so với Gaussian blur.

---

### Bước 4 – Phân đoạn & lọc contour

```python
def find_and_filter_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w
        area = cv2.contourArea(cnt)
        if 0.2 < aspect_ratio < 5.0 and area > 100 and h > 20:
            filtered.append((x, y, w, h))
    return filtered
```

---

### Bước 5 – Nhận dạng ký tự (EasyOCR)

```python
def recognize_characters(plate_image):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(plate_image)
    text = ''.join([t for (_, t, conf) in results if conf > 0.5])
    return text
```

---

### Bước 6 – Huấn luyện YOLOv8

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='license_plate.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    optimizer='AdamW',
    lr0=0.01,
    lrf=0.0001,
    device='cpu'
)
```

**Chiến lược learning rate:**

| Giai đoạn | Epochs | Learning Rate |
|---|---|---|
| Warm-up | 1 – 20 | `1e-2` |
| Tinh chỉnh | 21 – 80 | `1e-3` |
| Hội tụ | 81 – 100 | `1e-4` |

---

## 📊 Kết quả thực nghiệm

### So sánh với baseline (P. Surekha et al.)

| Chỉ số | P. Surekha (Baseline) | Ours (Proposed) | Cải thiện |
|---|---|---|---|
| 🎯 Tỷ lệ phát hiện | 65.40% | **70.69%** | ▲ +5.29% |
| 🌗 Độ tương phản | 1.05x | **1.28x** | ▲ +21.9% |
| 🔆 Độ sắc nét | 420.0 | **1492.64** | ▲ +3.55x |
| ⭐ Điểm chất lượng | 412.50 | **746.96** | ▲ +81.1% |
| ⏱️ Thời gian xử lý | 115 ms | 137 ms | ▼ +22ms |

> ✅ **92.2%** ảnh (107/116 mẫu) đạt chuẩn **Grade A** sau tiền xử lý.

### Thông số mô hình YOLO

- **Model:** YOLOv8n (Nano) — tối ưu cho CPU
- **Input size:** 640×640
- **mAP phát hiện:** > 97%
- **FPS (GPU):** > 30 FPS

### Hạn chế đã biết

- ❌ Ảnh bị nhòe do chuyển động (motion blur) mức cao
- ❌ Biển số bị che khuất > 30% diện tích

---

## 📐 Tiêu chuẩn biển số Việt Nam được hỗ trợ

| Loại | Kích thước | Tỉ lệ | Số ký tự |
|---|---|---|---|
| Biển dài (1 hàng) | 470 × 110 mm | 3.5 – 6.5 | 7 – 9 |
| Biển ngắn (2 hàng) | 280 × 200 mm | 0.8 – 1.5 | 7 – 9 |

---

## 🔭 Hướng phát triển

- [ ] Tích hợp **lightweight deep learning** để tự động điều chỉnh tham số tiền xử lý
- [ ] Bổ sung **motion deblurring** cho môi trường giao thông tốc độ cao
- [ ] Mở rộng dataset (điều kiện thời tiết, góc chụp đa dạng)
- [ ] Triển khai trên **thiết bị nhúng** (Raspberry Pi, Jetson Nano)
- [ ] Xây dựng **REST API** tích hợp hệ thống bãi đỗ xe

---

## 📚 Tài liệu tham khảo

1. P. Surekha, S. Sumathi – *"Automatic Number Plate Recognition Using Edge Detection and Morphological Operations"*, IJCA.
2. R. Gonzalez, R. Woods – *Digital Image Processing*, 4th ed., Pearson, 2018.
3. Ultralytics – *YOLOv8 Documentation*, 2023.
4. J. Redmon et al. – *"You Only Look Once: Unified, Real-Time Object Detection"*, CVPR, 2016.
5. Anagnostopoulos et al. – *"License Plate Recognition From Still Images and Video Sequences: A Survey"*, IEEE ITS, 2008.

---

## 👨‍💻 Nhóm thực hiện

| Họ tên | MSSV | Vai trò |
|---|---|---|
| Nguyễn Minh Năng | 0211668 | Xử lý ảnh & Pipeline |
| Đỗ Công Trí | 0214268 | Huấn luyện YOLO & Đánh giá |
| Nguyễn Đức Mạnh | 0210668 | OCR & Tích hợp hệ thống |

**Giảng viên hướng dẫn:** Thầy Đào Việt Cường  
**Trường:** Đại học Xây dựng Hà Nội – Khoa Công nghệ Thông tin  
**Năm:** 2025

---

## 📄 License

Dự án được phát hành theo giấy phép [MIT License](LICENSE).

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy để lại một Star!**

</div>  
