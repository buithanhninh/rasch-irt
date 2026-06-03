# rasch-irt 📊

`rasch-irt` là một thư viện Python chuyên nghiệp, hiệu năng cao phục vụ việc phân tích chất lượng câu hỏi thi trắc nghiệm và đánh giá năng lực thí sinh. Thư viện tích hợp đầy đủ hai phương pháp đo lường khảo thí hiện đại: **Lý thuyết Khảo thí Cổ điển (Classical Test Theory - CTT)** và **Lý thuyết Ứng đáp Câu hỏi (Item Response Theory - IRT)** với các mô hình 1PL (Rasch), 2PL, và 3PL sử dụng thuật toán **JMLE (Joint Maximum Likelihood Estimation)** Newton-Raphson tối ưu.

---

## 🇻🇳 TIẾNG VIỆT - HƯỚNG DẪN SỬ DỤNG CHI TIẾT

### 🚀 1. Cài đặt (Installation)
Thư viện yêu cầu Python >= 3.9 và các thư viện tính toán ma trận khoa học NumPy, SciPy. Cài đặt trực tiếp từ thư mục chứa thư viện:
```bash
pip install .
```

---

### 📝 2. Quy trình Sử dụng E2E (Chấm thi → CTT → IRT)

Hệ thống cung cấp một luồng dữ liệu tự nhiên và khép kín đi qua 3 bước:

#### Bước A: Chấm điểm tự động (`score_responses`)
Nhận ma trận phản hồi gốc (dạng chữ `'A', 'B'` hoặc dạng số `'1', '2'`) của thí sinh và đối chiếu với đáp án chuẩn (Answer Key) để xuất ra ma trận nhị phân `0/1` (1 = Đúng, 0 = Sai hoặc bỏ trống):

```python
import numpy as np
from rasch_irt import score_responses

# Phản hồi thô của 5 thí sinh làm bài thi 4 câu hỏi (A, B, C, D)
# Ký tự trống '' là học sinh bỏ qua không trả lời
raw_responses = np.array([
    ['A', 'B', 'A', 'D', ''],
    ['B', 'B', 'C', 'D', 'A'],
    ['C', 'A', 'D', 'C', 'C'],
    ['A', 'B', 'D', 'D', 'D']
])

# Đáp án chuẩn tương ứng cho 4 câu hỏi
answer_key = np.array(['A', 'B', 'D', 'D'])

# Chấm thi tự động
U = score_responses(raw_responses, answer_key)
print("Ma trận nhị phân chấm thi U:\n", U)
# Output: [4 câu hỏi x 5 thí sinh] nhị phân 0/1
```

#### Bước B: Phân tích Lý thuyết Cổ điển (`run_ctt`)
Đánh giá độ tin cậy của bài thi và lọc các câu hỏi lỗi trước khi chuyển sang mô hình toán học IRT phức tạp:

```python
from rasch_irt import run_ctt

ctt_result = run_ctt(U, raw_responses=raw_responses, answer_key=answer_key)

print(f"Độ tin cậy bài thi KR-20: {ctt_result.kr20}")
print(f"Hệ số Cronbach's Alpha: {ctt_result.cronbach_alpha}")
print(f"Điểm trung bình thô: {ctt_result.mean_score}")

# Quét và lấy các câu hỏi lỗi thô (Độ phân biệt D < 0 hoặc point-biserial tương quan âm)
print("Các câu hỏi kém chất lượng đề xuất loại bỏ:", ctt_result.bad_items)
```

#### Bước C: Ước lượng mô hình IRT 1PL / 2PL / 3PL (`run_irt`)
Sử dụng thuật toán ước lượng hợp lý cực đại đồng thời **JMLE Newton-Raphson** để tính toán năng lực thí sinh và tham số câu hỏi:

```python
from rasch_irt import run_irt, JMLEConfig

# 1. Cấu hình mô hình hiệu chỉnh (Calibration Config)
# Ví dụ: Chọn chạy mô hình 3PL cho đề trắc nghiệm 4 phương án lựa chọn
config = JMLEConfig(
    model_type=3,       # 1 = 1PL (Rasch), 2 = 2PL, 3 = 3PL
    num_options=4,      # Số phương án để tính Beta Prior cho tham số đoán mò c
    max_iter=100,       # Số vòng lặp JMLE tối đa
    tol=0.001           # Sai số epsilon hội tụ
)

# 2. Thực thi thuật toán IRT (loại trừ các câu hỏi thô bị lỗi tính toán)
# Lọc ma trận U sạch
excluded_indices = [idx - 1 for idx in ctt_result.bad_items]
U_clean = np.delete(U, excluded_indices, axis=0)

irt_result = run_irt(U_clean, config)

# 3. Đọc kết quả câu hỏi (Items Parameter)
for item in irt_result.items:
    print(f"Câu {item.item_number} -> Phân biệt (a): {item.param_a:.2f}, Độ khó (b): {item.param_b:.2f}, Đoán mò (c): {item.param_c:.2f}")
    print(f"   SE_b: {item.se_b:.3f}, Infit MNSQ: {item.infit_mnsq:.2f}, Outfit MNSQ: {item.outfit_mnsq:.2f} ({item.fit_flag})")

# 4. Đọc năng lực học sinh (Persons Abilities)
for person in irt_result.persons:
    print(f"Thí sinh index {person.student_code} -> Năng lực thực (theta): {person.theta:.2f}, Điểm quy đổi thang 10: {person.true_score_10:.1f}")
```

---

## 🇬🇧 ENGLISH - COMPLETE DOCUMENTATION

### 🚀 1. Installation
Install the packaged library easily via pip:
```bash
pip install .
```

---

### 📝 2. E2E Data Flow (Scoring → CTT → IRT)

#### Step A: Scoring responses (`score_responses`)
Converts raw alphabetic/numeric responses of students into a binary score matrix $U$ ($1$ for correct response, $0$ for incorrect or missing):
```python
import numpy as np
from rasch_irt import score_responses

raw_responses = np.array([
    ['A', 'B', 'A', 'D', ''],
    ['B', 'B', 'C', 'D', 'A'],
    ['C', 'A', 'D', 'C', 'C']
])
answer_key = np.array(['A', 'B', 'D'])

U = score_responses(raw_responses, answer_key)
```

#### Step B: Classical Test Theory (`run_ctt`)
Filters out poorly performing items (negative discrimination) and calculates baseline reliability (KR-20, Cronbach Alpha):
```python
from rasch_irt import run_ctt

ctt_result = run_ctt(U)
print("KR-20 Reliability:", ctt_result.kr20)
print("Items recommended for exclusion:", ctt_result.bad_items)
```

#### Step C: Item Response Theory (`run_irt`)
Performs dynamic parameters estimation under 1PL, 2PL, or 3PL models:
```python
from rasch_irt import run_irt, JMLEConfig

config = JMLEConfig(model_type=3, num_options=4)
irt_result = run_irt(U, config)

# Print item parameters
for item in irt_result.items:
    print(f"Item {item.item_number} - Difficulty (b): {item.param_b:.2f}, Discrimination (a): {item.param_a:.2f}")
```

---

## 🧮 3. Đặc tả Thuật toán & Toán học (Mathematical Specifications)

### A. Phương trình Mô hình 3PL (3PL Model Probability)
Xác suất thí sinh $j$ có năng lực $\theta_j$ trả lời đúng câu hỏi $i$ có tham số ($a_i, b_i, c_i$) là:
$$P_i(\theta_j) = c_i + \frac{1 - c_i}{1 + e^{-D \cdot a_i(\theta_j - b_i)}}$$
Trong đó:
*   $b_i$: Tham số độ khó câu hỏi (Difficulty)
*   $a_i$: Tham số độ phân biệt câu hỏi (Discrimination). Cố định bằng $1.0$ trong mô hình 1PL.
*   $c_i$: Tham số đoán mò ngẫu nhiên (Guessing). Cố định bằng $0.0$ trong mô hình 1PL/2PL.
*   $D = 1.702$: Hằng số scaling đưa mô hình Logistic tiệm cận mô hình Normal Ogive.

### B. Beta Prior cho Tham số Đoán mò $c$ (MAP Estimation)
Để tránh hiện tượng tham số đoán mò $c$ bị phân kỳ hoặc ước lượng bất hợp lý do cỡ mẫu nhỏ, thư viện áp dụng **Beta Prior** làm MAP estimator:
$$\text{Beta}(\alpha, \beta)$$
Các tham số Prior tự động ánh xạ theo số lượng đáp án lựa chọn (`num_options`) của bài thi:
*   4 phương án (A,B,C,D): $E[c] = 0.25 \rightarrow \alpha = 5.0, \beta = 15.0$
*   5 phương án: $E[c] = 0.20 \rightarrow \alpha = 4.0, \beta = 16.0$
*   3 phương án: $E[c] = 0.33 \rightarrow \alpha = 6.6, \beta = 13.4$

### C. Chỉ số Trùng khớp (Fit Statistics MNSQ)
Đánh giá mức độ khớp giữa dữ liệu thực nghiệm và mô hình toán học:
$$\text{OUTFIT MNSQ} = \frac{1}{M}\sum_{j=1}^M \frac{(u_{ij} - P_{ij})^2}{W_{ij}}$$
$$\text{INFIT MNSQ} = \frac{\sum_{j=1}^M (u_{ij} - P_{ij})^2}{\sum_{j=1}^M W_{ij}}$$
Trong đó $W_{ij} = P_{ij}(1 - P_{ij})$ là phương sai.
*   **MNSQ $\in [0.7, 1.3]$**: Câu hỏi chất lượng **Tốt** (Lý tưởng).
*   **MNSQ $> 1.3$**: Underfit (Dữ liệu bị nhiễu động cao hoặc có đoán mò diện rộng).
*   **MNSQ $< 0.7$**: Overfit (Câu hỏi quá dễ đoán hoặc trùng lặp dữ liệu).

---

## 📜 LICENSE
Bản quyền phân phối mã nguồn thuộc về tác giả **Bùi Thành Ninh** (MIT License).
Thư viện được đóng gói an toàn và sẵn sàng để public hoặc tích hợp chuyên nghiệp.
