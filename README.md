# Phân Tích Động Lực Nhu Cầu Khách Sạn và Tối Ưu Hóa Doanh Thu Dựa Trên Mô Hình Học Máy


<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.1-blue.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.17.0-purple.svg)](https://plotly.com/)

</div>

---

> **Tóm tắt:** Nghiên cứu này trình bày một phương pháp tiếp cận toàn diện trong việc phân tích hành vi hủy đặt phòng khách sạn và xây dựng mô hình tối ưu hóa doanh thu. Thông qua việc áp dụng các thuật toán học máy bao gồm Logistic Regression, Random Forest và Histogram-based Gradient Boosting, nghiên cứu đạt được hiệu năng dự đoán với PR-AUC đạt 0.7746 và ROC-AUC đạt 0.8547. Kết quả phân tích cho thấy việc áp dụng chiến lược giá tối ưu có thể tạo ra mức tăng doanh thu kỳ vọng lên đến 8-12% cho các phân khúc được ưu tiên, đồng thời cung cấp framework ra quyết định định lượng hỗ trợ quản lý doanh thu trong ngành khách sạn.

---

## Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
   - [1.1. Mô Tả Bài Toán](#11-mô-tả-bài-toán)
   - [1.2. Động Lực và Ứng Dụng Thực Tế](#12-động-lực-và-ứng-dụng-thực-tế)
   - [1.3. Mục Tiêu Nghiên Cứu](#13-mục-tiêu-nghiên-cứu)
2. [Dataset](#2-dataset)
   - [2.1. Nguồn Dữ Liệu](#21-nguồn-dữ-liệu)
   - [2.2. Mô Tả Các Features](#22-mô-tả-các-features)
   - [2.3. Kích Thước và Đặc Điểm Dữ Liệu](#23-kích-thước-và-đặc-điểm-dữ-liệu)
3. [Phương Pháp Nghiên Cứu](#3-phương-pháp-nghiên-cứu)
   - [3.1. Quy Trình Xử Lý Dữ Liệu](#31-quy-trình-xử-lý-dữ-liệu)
   - [3.2. Thuật Toán Sử Dụng](#32-thuật-toán-sử-dụng)
4. [Cài Đặt và Thiết Lập](#4-cài-đặt-và-thiết-lập)
5. [Hướng Dẫn Sử Dụng](#5-hướng-dẫn-sử-dụng)
6. [Kết Quả Nghiên Cứu](#6-kết-quả-nghiên-cứu)
   - [6.1. Kết Quả Đạt Được](#61-kết-quả-đạt-được)
   - [6.2. Trực Quan Hóa Kết Quả](#62-trực-quan-hóa-kết-quả)
   - [6.3. So Sánh và Phân Tích](#63-so-sánh-và-phân-tích)
7. [Cấu Trúc Dự Án](#7-cấu-trúc-dự-án)
8. [Khó Khăn và Giải Pháp](#8-khó-khăn-và-giải-pháp)
9. [Hướng Phát Triển](#9-hướng-phát-triển)
10. [Thông Tin Tác Giả](#10-thông-tin-tác-giả)
11. [Liên Hệ](#11-liên-hệ)
12. [Giấy Phép](#12-giấy-phép)
13. [Tài Liệu Tham Khảo](#13-tài-liệu-tham-khảo)

---

## 1. Giới Thiệu

### 1.1. Mô Tả Bài Toán

Nghiên cứu này tập trung vào bài toán **phân tích nhu cầu đặt phòng khách sạn và dự đoán hành vi hủy đặt phòng** (Hotel Booking Cancellation Prediction) - một vấn đề quan trọng trong lĩnh vực Revenue Management của ngành khách sạn. 

Bài toán này được phân loại thuộc lĩnh vực **Predictive Analytics** và **Revenue Management**, kết hợp các phương pháp phân tích thống kê mô tả, suy luận thống kê và mô hình hóa học máy.

### 1.2. Động Lực và Ứng Dụng Thực Tế

Ngành công nghiệp khách sạn toàn cầu phải đối mặt với thách thức nghiêm trọng về **tối ưu hóa doanh thu trong điều kiện nhu cầu bất định**. Theo nghiên cứu của Antonio et al. [1], tỷ lệ hủy đặt phòng trung bình trong ngành dao động từ 20-40%, gây ra tổn thất doanh thu đáng kể và khó khăn trong việc dự báo công suất.

**Các ứng dụng thực tế của nghiên cứu:**

| Lĩnh Vực | Ứng Dụng | Giá Trị Mang Lại |
|----------|----------|------------------|
| **Revenue Management** | Xác định mức giá tối ưu theo phân khúc | Tối đa hóa doanh thu thực nhận |
| **Risk Management** | Dự báo xác suất hủy để điều chỉnh chính sách | Giảm thiểu rủi ro phòng trống |
| **Marketing Strategy** | Nhận diện phân khúc khách hàng giá trị cao | Phân bổ nguồn lực marketing hiệu quả |
| **Operations** | Cải thiện dự báo công suất | Tối ưu lập kế hoạch nhân sự |

### 1.3. Mục Tiêu Nghiên Cứu

Nghiên cứu đặt ra các mục tiêu cụ thể theo framework SMART:

**Mục tiêu 1 - Khám Phá Dữ Liệu:** Phân tích phân phối và đặc điểm thống kê của các biến chính, nhận diện và xử lý các giá trị bất thường và dữ liệu thiếu, đánh giá chất lượng dữ liệu theo các tiêu chí completeness, accuracy và consistency.

**Mục tiêu 2 - Phân Tích Thăm Dò:** Phân tích ADR và tỷ lệ hủy theo các chiều phân khúc (loại khách sạn, thị trường, kênh phân phối), xác định các driver variables ảnh hưởng đến hành vi hủy đặt phòng, định lượng mối quan hệ đánh đổi giữa giá phòng và rủi ro hủy.

**Mục tiêu 3 - Mô Hình Hóa Dự Đoán:** Xây dựng và so sánh hiệu năng các thuật toán học máy, đạt ngưỡng hiệu năng ROC-AUC ≥ 0.80 và PR-AUC ≥ 0.70, xác định feature importance để hỗ trợ diễn giải kết quả.

**Mục tiêu 4 - Tối Ưu Hóa Giá:** Xây dựng framework tính Expected Realized Revenue, xác định mức ADR tối ưu cho từng phân khúc, ước lượng revenue uplift khi áp dụng chiến lược giá tối ưu.

**Mục tiêu 5 - Chuyển Đổi Insight Thành Hành Động:** Đề xuất lộ trình triển khai theo phase, xây dựng ma trận ưu tiên phân khúc, thiết lập các KPIs theo dõi hiệu quả triển khai.

---

## 2. Dataset

### 2.1. Nguồn Dữ Liệu

| Thuộc Tính | Thông Tin |
|------------|-----------|
| **Nền tảng** | Kaggle |
| **Tên bộ dữ liệu** | [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| **Tác giả gốc** | Nuno Antonio, Ana de Almeida, Luis Nunes |
| **Nguồn công bố** | *Data in Brief*, Volume 22, tháng 02/2019, trang 41-49 |
| **DOI** | [10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126) |
| **Giấy phép** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Thời gian thu thập** | Tháng 7/2015 đến tháng 8/2017 |
| **Nguồn gốc** | Property Management System (PMS) của hai khách sạn thực tế tại Bồ Đào Nha |

Dữ liệu được thu thập từ hệ thống quản lý tài sản của hai khách sạn thực tế tại Bồ Đào Nha: một **City Hotel** (khách sạn nội thành) và một **Resort Hotel** (khách sạn nghỉ dưỡng). Các thông tin nhạy cảm đã được ẩn danh hóa để bảo vệ quyền riêng tư.

### 2.2. Mô Tả Các Features

Bộ dữ liệu gồm **32 biến**, được phân loại theo các nhóm ngữ nghĩa:

#### **Nhóm A: Thông Tin Booking Cơ Bản**

| Feature | Mô Tả | Kiểu Dữ Liệu |
|---------|-------|--------------|
| `hotel` | Loại khách sạn (City Hotel / Resort Hotel) | Categorical |
| `is_canceled` | Trạng thái hủy đặt phòng (0 = không hủy, 1 = hủy) - **Biến mục tiêu** | Binary |
| `lead_time` | Số ngày từ lúc đặt phòng đến ngày nhận phòng | Integer |
| `arrival_date_year` | Năm đến | Integer |
| `arrival_date_month` | Tháng đến | Categorical |
| `arrival_date_week_number` | Số tuần trong năm | Integer |
| `arrival_date_day_of_month` | Ngày trong tháng | Integer |

#### **Nhóm B: Thông Tin Khách Hàng**

| Feature | Mô Tả | Kiểu Dữ Liệu |
|---------|-------|--------------|
| `adults` | Số lượng người lớn | Integer |
| `children` | Số lượng trẻ em | Integer |
| `babies` | Số lượng trẻ sơ sinh | Integer |
| `country` | Quốc gia của khách hàng (mã ISO 3166-1 alpha-3) | Categorical |
| `customer_type` | Loại khách hàng (Transient, Contract, Group, Transient-Party) | Categorical |
| `is_repeated_guest` | Khách quay lại (0 = không, 1 = có) | Binary |
| `previous_cancellations` | Số lần hủy trong lịch sử | Integer |
| `previous_bookings_not_canceled` | Số booking không hủy trong lịch sử | Integer |

#### **Nhóm C: Thông Tin Đặt Phòng**

| Feature | Mô Tả | Kiểu Dữ Liệu |
|---------|-------|--------------|
| `stays_in_weekend_nights` | Số đêm lưu trú cuối tuần | Integer |
| `stays_in_week_nights` | Số đêm lưu trú trong tuần | Integer |
| `meal` | Gói bữa ăn (BB, HB, FB, Undefined, SC) | Categorical |
| `reserved_room_type` | Loại phòng đặt | Categorical |
| `assigned_room_type` | Loại phòng thực tế được giao | Categorical |
| `booking_changes` | Số lần thay đổi booking | Integer |
| `deposit_type` | Loại đặt cọc (No Deposit, Non Refund, Refundable) | Categorical |
| `days_in_waiting_list` | Số ngày trong danh sách chờ | Integer |

#### **Nhóm D: Kênh Phân Phối và Thị Trường**

| Feature | Mô Tả | Kiểu Dữ Liệu |
|---------|-------|--------------|
| `market_segment` | Phân khúc thị trường (Online TA, Offline TA/TO, Direct, Corporate, Groups) | Categorical |
| `distribution_channel` | Kênh phân phối (TA/TO, Direct, Corporate, GDS) | Categorical |
| `agent` | ID đại lý đặt phòng | Categorical |
| `company` | ID công ty đặt phòng | Categorical |

#### **Nhóm E: Thông Tin Giá và Dịch Vụ**

| Feature | Mô Tả | Đơn Vị |
|---------|-------|--------|
| `adr` | Average Daily Rate - Giá phòng trung bình | EUR/đêm |
| `total_of_special_requests` | Tổng số yêu cầu đặc biệt | Integer |
| `required_car_parking_spaces` | Số chỗ đậu xe yêu cầu | Integer |

### 2.3. Kích Thước và Đặc Điểm Dữ Liệu

#### **Thống Kê Tổng Quan**

| Chỉ Số | Giá Trị |
|--------|---------|
| Số quan sát | 119,390 bookings |
| Số biến | 32 variables |
| Kích thước file | ~5 MB (CSV format) |
| Thời gian bao phủ | 2015-07 đến 2017-08 (25 tháng) |

#### **Phân Bố Biến Mục Tiêu**

| Trạng Thái | Số Lượng | Tỷ Lệ |
|------------|----------|-------|
| Không hủy (`is_canceled = 0`) | ~75,166 | 62.96% |
| Hủy (`is_canceled = 1`) | ~44,224 | 37.04% |

Dữ liệu có sự mất cân bằng nhẹ nhưng không quá nghiêm trọng, không cần áp dụng các kỹ thuật resampling phức tạp.

#### **Phân Bố Theo Loại Khách Sạn**

| Loại Khách Sạn | Số Bookings | Tỷ Lệ Hủy |
|----------------|-------------|-----------|
| City Hotel | ~79,330 | ~41.73% |
| Resort Hotel | ~40,060 | ~27.76% |

#### **Tình Trạng Dữ Liệu Thiếu**

| Biến | Số Quan Sát Thiếu | Tỷ Lệ | Chiến Lược Xử Lý |
|------|-------------------|-------|------------------|
| `children` | 4 | 0.003% | Impute bằng median |
| `country` | 488 | 0.41% | Gán nhãn "Unknown" |
| `agent` | 16,340 | 13.68% | Giữ nguyên (có ý nghĩa nghiệp vụ) |
| `company` | 112,593 | 94.31% | Giữ nguyên (có ý nghĩa nghiệp vụ) |

---

## 3. Phương Pháp Nghiên Cứu

### 3.1. Quy Trình Xử Lý Dữ Liệu

Quy trình xử lý dữ liệu được thực hiện theo methodology CRISP-DM (Cross-Industry Standard Process for Data Mining), bao gồm các bước sau:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Loading   │───▶│  Data Cleaning  │───▶│ Feature Eng.    │───▶│  Data Splitting │
│                 │    │                 │    │                 │    │                 │
│ • Load CSV      │    │ • Handle NaN    │    │ • Create vars   │    │ • Train/Test    │
│ • Initial check │    │ • Remove outlier│    │ • Encoding      │    │ • Stratify      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
                                                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Deployment    │◀───│   Evaluation    │◀───│    Modeling     │◀───│  Preprocessing  │
│                 │    │                 │    │                 │    │                 │
│ • Optimization  │    │ • Metrics       │    │ • Train models  │    │ • Scaling       │
│ • Prioritize    │    │ • Comparison    │    │ • Tune params   │    │ • Transform     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### **Bước 1: Làm Sạch Dữ Liệu (Data Cleaning)**

**Xử lý giá trị thiếu (Missing Values):**

Chiến lược xử lý được thiết kế dựa trên đặc điểm nghiệp vụ của từng biến:
- **Biến `children`:** Phân phối lệch phải nên sử dụng median để impute, đảm bảo giá trị thay thế không bị ảnh hưởng bởi các giá trị cực đoan.
- **Biến `country`:** Không xác định được nguồn gốc khách hàng nên gán nhãn "Unknown" để giữ lại thông tin về sự thiếu hụt này.
- **Biến `agent` và `company`:** Giá trị NaN có ý nghĩa nghiệp vụ (booking trực tiếp không qua đại lý/công ty) nên giữ nguyên.

**Xử lý giá trị bất thường (Outliers):**

Áp dụng phương pháp **Winsorization** để xử lý outliers trong biến ADR. Phương pháp này giới hạn các giá trị cực đoan tại percentile 1 và 99, giúp giảm ảnh hưởng của outliers mà không loại bỏ hoàn toàn các quan sát.

$$x_{winsorized} = \begin{cases} P_{1} & \text{if } x < P_{1} \\ x & \text{if } P_{1} \leq x \leq P_{99} \\ P_{99} & \text{if } x > P_{99} \end{cases}$$

**Loại bỏ quan sát không hợp lệ:**
- ADR âm (không hợp lệ về mặt nghiệp vụ)
- Booking không có khách (tổng adults + children + babies = 0)

**Loại bỏ biến gây Data Leakage:**

Các biến `reservation_status` và `reservation_status_date` chỉ biết được sau khi sự kiện xảy ra, do đó cần loại bỏ để tránh data leakage trong quá trình modeling.

#### **Bước 2: Feature Engineering**

**Tạo biến dẫn xuất (Derived Features):**
- `total_nights`: Tổng số đêm lưu trú = stays_in_weekend_nights + stays_in_week_nights
- `total_guests`: Tổng số khách = adults + children + babies
- `has_children`: Biến nhị phân đánh dấu booking có trẻ em
- `lead_time_category`: Phân nhóm lead time thành 5 categories (0-7d, 8-30d, 31-90d, 91-180d, 180+d)

**Encoding biến phân loại:**

Sử dụng **One-Hot Encoding** cho các biến categorical nominal (hotel, meal, market_segment, distribution_channel, deposit_type, customer_type) với drop_first=True để tránh multicollinearity.

#### **Bước 3: Chia Tách Dữ Liệu**

Dữ liệu được chia theo tỷ lệ 80/20 (training/test) với **stratified sampling** để đảm bảo tỷ lệ class được duy trì trong cả hai tập.

#### **Bước 4: Chuẩn Hóa Features**

Áp dụng **Z-score standardization** cho các biến số. Quan trọng là chỉ fit scaler trên training set và transform cả training và test set để tránh data leakage.

### 3.2. Thuật Toán Sử Dụng

Nghiên cứu triển khai và so sánh bốn thuật toán học máy:

#### **A. Logistic Regression**

Logistic Regression mô hình hóa xác suất hủy đặt phòng thông qua hàm logistic (sigmoid):

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**Ưu điểm:**
- Khả năng diễn giải cao thông qua odds ratio
- Training nhanh
- Không yêu cầu tuning nhiều hyperparameters

**Nhược điểm:**
- Giả định mối quan hệ tuyến tính giữa features và log-odds
- Không capture được các interaction phức tạp

#### **B. Random Forest**

**Ưu điểm:**
- Xử lý tốt các mối quan hệ phi tuyến
- Robust với outliers
- Cung cấp feature importance

**Nhược điểm:**
- Training chậm hơn Logistic Regression
- Kém interpretable hơn

#### **C. Gradient Boosting**

**Ưu điểm:**
- Hiệu năng cao
- Xử lý tốt các mối quan hệ phức tạp

**Nhược điểm:**
- Training rất chậm
- Dễ overfitting nếu không tuning đúng

#### **D. Histogram-based Gradient Boosting (HistGradientBoosting)**

Phiên bản tối ưu của Gradient Boosting, discretize continuous features vào bins và xây dựng histogram để tăng tốc độ training.

**Ưu điểm:**
- Hiệu năng cao nhất trong các thuật toán được thử nghiệm
- Training nhanh hơn nhiều so với Gradient Boosting thông thường
- Xử lý tốt missing values native

**Đây là thuật toán được chọn làm mô hình chính** vì cân bằng tốt giữa accuracy và speed.

#### **E. Phương Pháp Tối Ưu Hóa ADR**

Nghiên cứu phát triển framework tính **Expected Realized Revenue** để xác định mức giá tối ưu:

$$E[Revenue] = ADR \times nights \times (1 - P(Cancel|ADR, \mathbf{x}))$$

Phương pháp này cân bằng giữa việc tăng giá (tăng revenue per booking) và giảm rủi ro hủy (tăng probability of completion). Điểm tối ưu được tìm bằng cách mô phỏng các mức ADR khác nhau và dự đoán xác suất hủy tương ứng.

---

## 4. Cài Đặt và Thiết Lập

### 4.1. Yêu Cầu Hệ Thống

| Thành Phần | Yêu Cầu Tối Thiểu | Khuyến Nghị |
|------------|-------------------|-------------|
| **Hệ điều hành** | Windows 10, macOS 10.14, Ubuntu 18.04 | Ubuntu 20.04+ |
| **Python** | 3.9+ | 3.11 |

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/b28dbca4/Hotel-Demand-Dynamics.git
   cd Hotel-Demand-Dynamics
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ds-env
   ```

   Alternatively, install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

### 4.2. Các Thư Viện Chính

| Thư Viện | Phiên Bản | Mục Đích |
|----------|-----------|----------|
| `numpy` | 1.26.0 | Tính toán ma trận và vector |
| `pandas` | 2.1.1 | Xử lý và phân tích dữ liệu |
| `scikit-learn` | 1.3.1 | Machine learning algorithms |
| `matplotlib` | 3.8.0 | Trực quan hóa cơ bản |
| `seaborn` | 0.13.0 | Trực quan hóa thống kê |
| `plotly` | 5.17.0 | Biểu đồ tương tác |

### 4.3. Hướng Dẫn Cài Đặt

#### **Phương Án 1: Sử Dụng Conda (Khuyến nghị)**

1. Clone repository
2. Tạo môi trường conda từ file `environment.yml`
3. Kích hoạt môi trường `ds-env`
4. Xác nhận cài đặt thành công

#### **Phương Án 2: Sử Dụng pip**

1. Clone repository
2. Tạo virtual environment
3. Kích hoạt virtual environment
4. Cài đặt dependencies từ `requirements.txt`

### 4.4. Chuẩn Bị Dữ Liệu

Tải dữ liệu từ Kaggle và đặt vào thư mục `data/raw/hotel_bookings.csv`.

---

## 5. Hướng Dẫn Sử Dụng

### 5.1. Quy Trình Chạy Notebooks

Các notebooks được thiết kế để chạy tuần tự theo thứ tự:

```
01_data_explorations.ipynb → 02_data_preprocessing.ipynb → 03_eda_business.ipynb → 04_eda_operations_and_modeling.ipynb
```

### 5.2. Chi Tiết Từng Notebook

#### **Notebook 1: Khám Phá Dữ Liệu (`01_data_explorations.ipynb`)**

| Mục | Nội Dung | Output |
|-----|----------|--------|
| 1.1 | Tải và kiểm tra dữ liệu thô | Data summary |
| 1.2 | Phân tích biến mục tiêu (is_canceled) | Distribution plots |
| 1.3 | Phân tích biến ADR | Histogram, boxplot |
| 1.4 | Kiểm tra missing values | Missing value report |
| 1.5 | Phát hiện outliers | Outlier analysis |

#### **Notebook 2: Tiền Xử Lý Dữ Liệu (`02_data_preprocessing.ipynb`)**

| Mục | Nội Dung | Output |
|-----|----------|--------|
| 2.1 | Xử lý missing values | Imputation log |
| 2.2 | Xử lý outliers | Clipping summary |
| 2.3 | Loại bỏ quan sát bất hợp lệ | Filter report |
| 2.4 | Feature engineering | New features list |
| 2.5 | Export dữ liệu | `clean_data.csv` |

#### **Notebook 3: Phân Tích Kinh Doanh (`03_eda_business.ipynb`)**

| Mục | Nội Dung | Output |
|-----|----------|--------|
| 3.1 | ADR theo loại khách sạn | Comparison charts |
| 3.2 | Tỷ lệ hủy theo phân khúc | Heatmaps |
| 3.3 | ADR-Cancel relationship | Scatter plots |
| 3.4 | Segment value analysis | Strategic matrix |

#### **Notebook 4: Phân Tích Vận Hành và Mô Hình (`04_eda_operations_and_modeling.ipynb`)**

| Mục | Nội Dung | Output |
|-----|----------|--------|
| 4.1 | Driver analysis | Feature importance |
| 4.2 | ADR-Cancellation trade-off | Trade-off curves |
| 4.3 | Model training & evaluation | Performance metrics |
| 4.4 | ADR optimization | Optimal ADR recommendations |
| 4.5 | Impact simulation | Revenue uplift estimates |
| 4.6 | Segment prioritization | Rollout roadmap |

---

## 6. Kết Quả Nghiên Cứu

### 6.1. Kết Quả Đạt Được

#### **A. Hiệu Năng Mô Hình Dự Đoán**

| Mô Hình | ROC-AUC | PR-AUC | F1-Score | Training Time |
|---------|---------|--------|----------|---------------|
| **HistGradientBoosting** | **0.8547** | **0.7746** | **0.7023** | 2.67s |
| Random Forest | 0.8438 | 0.7605 | 0.6890 | 34.58s |
| Gradient Boosting | 0.8492 | 0.7678 | 0.6952 | 90.91s |
| Logistic Regression | 0.7981 | 0.6892 | 0.6234 | 0.42s |

**HistGradientBoostingClassifier** đạt hiệu năng tốt nhất với thời gian training nhanh, được chọn làm mô hình chính.

#### **B. Confusion Matrix (HistGradientBoosting)**

|  | Predicted: Not Cancel | Predicted: Cancel |
|--|----------------------|-------------------|
| **Actual: Not Cancel** | 12,847 (TN) | 2,186 (FP) |
| **Actual: Cancel** | 3,125 (FN) | 5,720 (TP) |

- **Precision**: 72.34%
- **Recall**: 64.68%
- **Specificity**: 85.46%

#### **C. Feature Importance (Top 10)**

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | `deposit_type_Non_Refund` | 0.2847 | Chính sách đặt cọc không hoàn tiền giảm mạnh tỷ lệ hủy |
| 2 | `lead_time` | 0.1523 | Lead time dài → xác suất hủy cao hơn |
| 3 | `adr` | 0.0892 | Giá cao → tăng rủi ro hủy |
| 4 | `total_of_special_requests` | 0.0654 | Nhiều yêu cầu đặc biệt → ít hủy hơn |
| 5 | `previous_cancellations` | 0.0587 | Lịch sử hủy dự báo hành vi tương lai |
| 6 | `market_segment_Online_TA` | 0.0512 | Kênh Online TA có tỷ lệ hủy cao |
| 7 | `is_repeated_guest` | 0.0423 | Khách quay lại ít hủy hơn |
| 8 | `booking_changes` | 0.0389 | Thay đổi booking → cam kết cao hơn |
| 9 | `customer_type_Transient` | 0.0356 | Khách lẻ có rủi ro hủy cao nhất |
| 10 | `days_in_waiting_list` | 0.0298 | Booking từ waitlist ít hủy hơn |

#### **D. Kết Quả Tối Ưu Hóa ADR**

| Loại Khách Sạn | ADR Hiện Tại | ADR Tối Ưu | Δ ADR | Expected Revenue Uplift |
|----------------|--------------|------------|-------|-------------------------|
| **City Hotel** | 105.30 EUR | 94.50 EUR | -10.8 EUR | +8.2% |
| **Resort Hotel** | 94.95 EUR | 89.20 EUR | -5.75 EUR | +5.6% |

Việc giảm ADR hợp lý giúp tăng expected revenue nhờ giảm tỷ lệ hủy.

### 6.2. Trực Quan Hóa Kết Quả

#### **A. Mối Quan Hệ ADR vs Cancel Rate**

Phân tích cho thấy mối quan hệ tích cực giữa ADR và tỷ lệ hủy: khi giá phòng tăng, xác suất khách hủy cũng tăng theo. City Hotel có độ dốc cao hơn Resort Hotel, cho thấy khách hàng của City Hotel nhạy cảm với giá hơn.

#### **B. Ma Trận Ưu Tiên Phân Khúc**

Các phân khúc được phân loại thành 4 nhóm:

| Quadrant | Đặc Điểm | Phân Khúc | Chiến Lược |
|----------|----------|-----------|------------|
| **Phase 1 (Win-Win)** | Revenue Uplift cao, Risk thấp | Online TA, Direct | Triển khai ngay |
| **Phase 2 (Quick Wins)** | Revenue Uplift trung bình, Risk thấp | Offline TA/TO, Transient | Triển khai tiếp theo |
| **Monitor** | Revenue Uplift cao, Risk cao | Groups, Corporate | Cần theo dõi kỹ |
| **Hold** | Revenue Uplift thấp | Undefined | Tạm hoãn |

#### **C. Mô Phỏng Tác Động Doanh Thu**

| Phase | Phân Khúc | Bookings | Revenue Uplift | Risk Level |
|-------|-----------|----------|----------------|------------|
| **Phase 1** | Online TA (City) | 28,500 | +12.3% | Low |
| **Phase 1** | Direct (City) | 8,200 | +9.8% | Low |
| **Phase 2** | Offline TA/TO | 15,600 | +6.5% | Medium |
| **Phase 2** | Transient (Resort) | 12,400 | +5.2% | Medium |

### 6.3. So Sánh và Phân Tích

#### **A. So Sánh Giữa Các Mô Hình**

| Tiêu Chí | Logistic Regression | Random Forest | HistGradientBoosting |
|----------|---------------------|---------------|----------------------|
| **Accuracy** | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **Interpretability** | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| **Training Speed** | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| **Handles Missing** | ★★☆☆☆ | ★★★☆☆ | ★★★★★ |

#### **B. So Sánh City Hotel vs Resort Hotel**

| Chỉ Số | City Hotel | Resort Hotel | Chênh Lệch |
|--------|------------|--------------|------------|
| Tỷ lệ hủy trung bình | 41.73% | 27.76% | +13.97% |
| ADR trung bình | 105.30 EUR | 94.95 EUR | +10.35 EUR |
| Lead time trung bình | 109 ngày | 92 ngày | +17 ngày |
| Potential Uplift | +8.2% | +5.6% | +2.6% |

**Insights chính:**
- City Hotel có tỷ lệ hủy cao hơn đáng kể (+14%)
- Giá phòng City Hotel cao hơn nhưng rủi ro cũng cao hơn
- Tiềm năng cải thiện ở City Hotel lớn hơn

---

## 7. Cấu Trúc Dự Án

```
Hotel-Demand-Dynamics/
│
├── data/                          # Thư mục dữ liệu
│   ├── raw/                       # Dữ liệu thô (hotel_bookings.csv)
│   ├── processed/                 # Dữ liệu đã xử lý (clean_data.csv)
│   └── final/                     # Dữ liệu cuối cùng cho modeling
│
├── notebooks/                        # Jupyter Notebooks phân tích
│   ├── 01_data_explorations.ipynb    # Khám phá dữ liệu ban đầu
│   ├── 02_data_preprocessing.ipynb   # Tiền xử lý và làm sạch
│   ├── 03_eda_business.ipynb         # Phân tích insights kinh doanh
│   └── 04_eda_operations_and_modeling.ipynb    # Modeling và tối ưu hóa
│
├── src/                           # Source code Python
│   ├── data/                      # Module xử lý dữ liệu
│   │   └── data_loader.py         # Functions tải dữ liệu
│   │
│   └── utils/                     # Utilities chung
│       └── data_quality.py        # Kiểm tra chất lượng dữ liệu
│
├── reports/                       # Báo cáo và outputs
├── environment.yml                # Conda environment file
├── requirements.txt               # pip requirements
├── LICENSE                        # Apache 2.0 License
└── README.md                      # Tài liệu này
```

---

## 8. Khó Khăn và Giải Pháp

### 8.1. Các Khó Khăn Gặp Phải

| # | Khó Khăn | Mô Tả Chi Tiết |
|---|----------|----------------|
| 1 | **Data Leakage** | Các biến `reservation_status` chứa thông tin về tương lai, có thể làm sai lệch kết quả mô hình |
| 2 | **Imbalanced Classes** | Tỷ lệ hủy 37% vs không hủy 63% có thể ảnh hưởng đến hiệu năng mô hình |
| 3 | **High Cardinality Categorical** | Biến `country` có 178 unique values, `agent` có 334 values gây khó khăn cho encoding |
| 4 | **Missing Values có ý nghĩa** | NaN trong `agent`/`company` không phải thiếu mà có ý nghĩa nghiệp vụ |
| 5 | **Trade-off giữa Accuracy và Interpretability** | Mô hình phức tạp cho accuracy cao nhưng khó giải thích cho stakeholders |
| 6 | **Outliers trong ADR** | Một số booking có ADR cực cao (>500 EUR) hoặc âm |

### 8.2. Giải Pháp Áp Dụng

| Khó Khăn | Giải Pháp | Kết Quả |
|----------|-----------|---------|
| **Data Leakage** | Loại bỏ các biến chỉ biết sau sự kiện (reservation_status, reservation_status_date) | Mô hình dự đoán đáng tin cậy hơn |
| **Imbalanced Classes** | Sử dụng stratified sampling và metrics phù hợp (PR-AUC thay vì chỉ accuracy) | Đánh giá chính xác hơn hiệu năng trên minority class |
| **High Cardinality** | Merge rare categories (<1%) vào "Other", giữ lại top N categories quan trọng | Giảm dimensionality mà không mất thông tin quan trọng |
| **Missing có ý nghĩa** | Giữ nguyên NaN và để model xử lý (HistGradientBoosting hỗ trợ native) | Bảo toàn semantic của dữ liệu |
| **Accuracy vs Interpretability** | Kết hợp HistGradientBoosting (accuracy) với Logistic Regression (interpretability) để có cả hai góc nhìn | Stakeholders hiểu được insights từ Logistic Regression, đồng thời có mô hình production mạnh mẽ |
| **Outliers** | Winsorization tại P1-P99 thay vì loại bỏ hoàn toàn | Giữ lại dữ liệu hữu ích, giảm ảnh hưởng của extreme values |

### 8.3. Bài Học Kinh Nghiệm

1. **Domain Knowledge là quan trọng:** Hiểu nghiệp vụ khách sạn giúp xác định đúng cách xử lý missing values và feature engineering.

2. **Không nên tối ưu hóa một metric duy nhất:** Cần cân bằng giữa nhiều metrics (precision, recall, AUC) tùy thuộc vào business objective.

3. **Interpretability quan trọng không kém Accuracy:** Trong bối cảnh business, khả năng giải thích kết quả cho non-technical stakeholders rất quan trọng.

4. **Validation strategy cần match với use case:** Sử dụng time-based split nếu mô hình sẽ được dùng để dự đoán tương lai.

---

## 9. Hướng Phát Triển


| Cải Tiến | Mô Tả | 
|----------|-------|
| Hyperparameter Tuning | Sử dụng Optuna/GridSearchCV để tối ưu hyperparameters | 
| Cross-validation | Thêm k-fold CV để đánh giá robust hơn | 
| Feature Selection | Áp dụng RFE, SHAP-based selection |
| Error Analysis | Phân tích chi tiết các trường hợp dự đoán sai | 

---

## 10. Thông Tin Tác Giả:

<table border="1" cellspacing="0" cellpadding="8" align="center">
  <tr>
    <th>MSSV</th>
    <th>Họ và Tên</th>
  </tr>
  <tr>
    <td>23120265</td>
    <td>Nguyễn Thái Hoàng</td>
  </tr>
  <tr>
    <td>23120348</td>
    <td>Ngô Thị Thục Quyên</td>
  </tr>
  <tr>
    <td>23120415</td>
    <td>Lăng Phú Quý</td>
  </tr>
</table>


### Thông Tin Khóa Học

| Thuộc Tính | Thông Tin |
|------------|-----------|
| **Môn học** | Programming for Data Science |
| **Mã môn học** | CSC17104 |
| **Học kỳ** | Học kì 1/2025 |

---

## 11. Liên Hệ

Nếu có câu hỏi hoặc góp ý về dự án, vui lòng liên hệ:

- **Email**: 

   - 23120265@student.hcmus.edu.vn

   - 23120384@student.hcmus.edu.vn 

   - 23120415@student.hcmus.edu.vn
 
---

## 12. Giấy Phép

Dự án này được phát hành dưới giấy phép **Apache License 2.0**.

Dữ liệu gốc từ Kaggle được phát hành dưới **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

---

## 13. Tài Liệu Tham Khảo

[1] N. Antonio, A. de Almeida, and L. Nunes, "Hotel booking demand datasets," *Data in Brief*, vol. 22, pp. 41–49, Feb. 2019. DOI: [Hotel booking demand datasets](https://doi.org/10.1016/j.dib.2018.11.126)

[2] N. Antonio, A. de Almeida, and L. Nunes, "Predicting hotel booking cancellation to decrease uncertainty and increase revenue," *Tourism & Management Studies*, vol. 13, no. 2, pp. 25–39, 2017. 

[3] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794. [A Scalabel Tree Boosting System](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

[4] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in Neural Information Processing Systems*, vol. 30, 2017. [A Hightly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

[5] Scikit-learn Documentation. [Online](https://scikit-learn.org/stable/)

[6] A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd ed. O'Reilly Media, 2019. [Online](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---

<div align="center">

**Cảm ơn bạn đã quan tâm đến dự án!**

*Nếu dự án hữu ích, hãy ⭐ star repository này!*

---

*Cập nhật lần cuối: Tháng 12, 2024*

</div>



