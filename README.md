# Project 5 - Machine Learning

## Giới thiệu

Trong project này, nhóm em thực hiện các bài tập về mô hình học máy bằng
Python và PyTorch. Nội dung gồm Perceptron, hồi quy hàm sin, phân loại chữ số
MNIST, nhận diện ngôn ngữ, convolution và cơ chế causal attention.

Dữ liệu dùng cho các bài phân loại nằm trong thư mục `data/`. Ngoài ra, project
có phần mở rộng Character GPT trong `gpt_model.py` và `chargpt.py`; để chạy
`chargpt.py` cần chuẩn bị thêm file `input.txt`.

## Cách cài đặt

Cài Python 3, sau đó mở terminal tại thư mục project và cài các thư viện:

```bash
python -m pip install numpy matplotlib torch
```

## Chạy lại chương trình

Để chạy autograder cho cả sáu câu:

```bash
python autograder.py --no-graphics
```

Kết quả đạt được sẽ được in ra terminal. Tùy chọn `--no-graphics` giúp chạy
kiểm tra mà không mở các cửa sổ biểu đồ.

Muốn huấn luyện lại và lưu ảnh kết quả cho các câu có biểu đồ, chạy:

```bash
python generate_results.py
```

Ảnh `question-1-result.png` đến `question-5-result.png` sẽ được lưu ngay trong
thư mục project. Script này huấn luyện lại các mô hình nên có thể mất vài
phút. Câu 6 được kiểm tra bằng autograder, không tạo ảnh kết quả.

## Một số file chính

- `models.py`: các mô hình dùng trong project.
- `train.py`: các hàm huấn luyện.
- `losses.py`: các hàm tính loss.
- `backend.py`: dữ liệu và các tiện ích hỗ trợ chạy bài.
- `autograder.py`: chạy kiểm tra và chấm các câu.
- `generate_results.py`: chạy mô hình và lưu ảnh kết quả.

## GitHub

Mã nguồn của project:
[GiuaKy-TriTueNhanTao/Project-5-Machine-Learning](https://github.com/GiuaKy-TriTueNhanTao/Project-5-Machine-Learning)

