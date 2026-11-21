# Project 5 - Machine Learning

## 1. Giới thiệu

Đây là project môn học về Machine Learning của nhóm sinh viên. Trong project,
nhóm triển khai và đánh giá sáu bài toán bằng Python và thư viện PyTorch.

Nội dung bao gồm các kiến thức cơ bản và nâng cao như perceptron, hồi quy,
mạng nơ-ron phân loại, recurrent neural network, convolution và causal
attention.

README và kết quả thực nghiệm chỉ đề cập phạm vi câu 1 đến câu 6.

## 2. Mục tiêu

- Hiểu cách xây dựng và huấn luyện mô hình học máy bằng PyTorch.
- Cài đặt các thành phần học máy theo yêu cầu của bài tập.
- Phân tích kết quả qua loss, validation accuracy và test accuracy.
- Trình bày kết quả bằng confusion matrix, biểu đồ hồi quy và demo trực quan.

## 3. Nội dung thực hiện

| Câu | Nội dung | Phương pháp chính |
|---|---|---|
| 1 | Perceptron phân loại hai lớp | Cập nhật trọng số bằng perceptron learning rule |
| 2 | Hồi quy hàm sin | Mạng nơ-ron hồi quy và mean squared error |
| 3 | Phân loại chữ số MNIST | Fully connected neural network |
| 4 | Nhận diện ngôn ngữ | Recurrent neural network xử lý chuỗi ký tự |
| 5 | Phân loại MNIST bằng convolution | Tự triển khai convolution với `tensordot` |
| 6 | Causal attention | Query, key, value và causal mask |

Dữ liệu huấn luyện và kiểm tra được lưu trong thư mục `data/`.

## 4. Công nghệ sử dụng

- Python
- PyTorch
- NumPy
- Matplotlib

## 5. Cấu trúc project

```text
.
├── autograder.py
├── backend.py
├── losses.py
├── models.py
├── train.py
├── data/
│   ├── mnist.npz
│   └── lang_id.npz
└── experiment_results/
    ├── metrics.csv
    ├── question-1.png
    ├── question-2.png
    ├── question-3.png
    ├── question-3-validation-visualization.png
    ├── question-4.png
    ├── question-5.png
    ├── question-5-validation-visualization.png
    └── question-6.png
```

Vai trò của các file chính:

- `models.py`: định nghĩa các mô hình của câu 1–6.
- `train.py`: triển khai quy trình huấn luyện.
- `losses.py`: định nghĩa các hàm loss.
- `backend.py`: nạp dữ liệu, chia tập dev/test và hỗ trợ tính accuracy.
- `autograder.py`: kiểm tra và chấm public test cho toàn bộ project.

## 6. Cài đặt môi trường

Cài Python 3, sau đó mở terminal tại thư mục project và chạy:

```bash
python -m pip install numpy matplotlib torch
```

Nếu đang sử dụng môi trường ảo `.venv` trên Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib torch
```

## 7. Hướng dẫn chạy

### 7.1. Chạy toàn bộ bài kiểm tra

```bash
python autograder.py --no-graphics
```

### 7.2. Chạy riêng từng câu

```bash
python autograder.py --no-graphics -q q1
python autograder.py --no-graphics -q q2
python autograder.py --no-graphics -q q3
python autograder.py --no-graphics -q q4
python autograder.py --no-graphics -q q5
python autograder.py --no-graphics -q q6
```

Nếu dùng môi trường ảo trên Windows:

```powershell
.\.venv\Scripts\python.exe autograder.py --no-graphics
```

Trong kết quả chạy, dòng `PASS` cho biết bài kiểm tra tương ứng đã đạt.
Bảng điểm tạm thời được hiển thị sau khi toàn bộ các bài kiểm tra kết thúc.

## 8. Kết quả thực nghiệm

Accuracy được tính bằng tỷ lệ mẫu dự đoán đúng trên tập tương ứng.
Các kết quả dưới đây được ghi lại sau quá trình huấn luyện thực tế.

| Câu | Validation accuracy | Test accuracy | Nhận xét |
|---|---:|---:|---|
| 1 - Perceptron | 99.80% | 99.70% | Accuracy trên hai tập holdout tổng hợp; tập train đạt 100% |
| 2 - Hồi quy sin | MSE 0.013284 | MSE 0.013236 | Bài này đánh giá bằng MSE, không dùng accuracy |
| 3 - MNIST | 98.02% | 97.42% | Đánh giá trên 5.000 dev và 5.000 test mẫu |
| 4 - Nhận diện ngôn ngữ | 85.60% | 85.40% | Mô hình đạt yêu cầu đề bài trên dev và test |
| 5 - Convolution | 100.00% | 100.00% | Tập nhỏ 50 mẫu/dev/test và có dữ liệu trùng với tập train |
| 6 - Causal attention | Không có | Không có | Kiểm tra tensor đầu ra; không phải bài phân loại |

Accuracy có thể thay đổi nhẹ khi huấn luyện lại do tính ngẫu nhiên của mô hình
và quá trình khởi tạo trọng số.

## 9. Minh họa kết quả

- [Câu 1 - Perceptron](experiment_results/question-1.png)
- [Câu 2 - Hồi quy sin](experiment_results/question-2.png)
- [Câu 3 - MNIST và confusion matrix](experiment_results/question-3.png)
- [Câu 3 - Demo xác suất theo từng chữ số](experiment_results/question-3-validation-visualization.png)
- [Câu 4 - Nhận diện ngôn ngữ](experiment_results/question-4.png)
- [Câu 5 - Convolution và confusion matrix](experiment_results/question-5.png)
- [Câu 5 - Demo xác suất theo từng chữ số](experiment_results/question-5-validation-visualization.png)
- [Câu 6 - Causal attention weights](experiment_results/question-6.png)
- [Bảng metrics](experiment_results/metrics.csv)

Biểu đồ demo câu 3 và câu 5 trình bày từng nhãn thật theo hàng. Trục ngang là
xác suất của nhãn đúng:

- Màu xanh: mô hình dự đoán đúng.
- Màu đỏ: mô hình dự đoán sai.
- Số nhỏ dưới ảnh đỏ: nhãn mô hình dự đoán.

Lưu ý: tập dữ liệu câu 5 có kích thước nhỏ và một phần dữ liệu dev/test trùng
hoặc nằm trong nguồn dữ liệu huấn luyện. Vì vậy, kết quả 100% cần được diễn
giải trong ngữ cảnh tập kiểm tra đặc thù của đề bài.

## 10. Kết luận

Nhóm đã hoàn thành triển khai và kiểm tra sáu câu thuộc phạm vi project:
Perceptron, hồi quy, phân loại MNIST, nhận diện ngôn ngữ, convolution và causal
attention. Các mô hình phân loại đạt kết quả phù hợp với ngưỡng của autograder;
kết quả được lưu kèm biểu đồ để thuận tiện đối chiếu và trình bày.

Hướng phát triển tiếp theo có thể là huấn luyện nhiều seed hơn, sử dụng tập
kiểm tra độc lập hơn và so sánh thêm các kiến trúc phân loại.

## GitHub

Mã nguồn của project:
[GiuaKy-TriTueNhanTao/Project-5-Machine-Learning](https://github.com/GiuaKy-TriTueNhanTao/Project-5-Machine-Learning)

