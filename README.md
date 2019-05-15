# Predict student's grade 
Đồ án môn khai thác dữ liệu
### Install
```bash
pip install -r requirements.txt
```
### Usage
* Train model theo train size đưa vào, có thể để trống, giá trị mặc định là 0.7
```bash
python program.py -t
```
* Dự đoán điểm dựa trên mẫu đưa vào, có thể chọn model dự đoán linear hoặc random forest, mặc định là linear
* Mâu đưa vào có dạng G2,G1,failures,schoolsup,absences,mother_edu trong đó
* G2 là điểm học kỳ 2 (từ 1 - 20)
* G1 là điểm học kỳ 1 (từ )
* failures là số môn học lại(1 - 4)
* schoolsup là có học phụ đạo hay không(2 giá trị 1: có, 0: không)
* absences là số buổi nghỉ học(0 - 93)
* mother_edu là trình độ học vấn của mẹ
(numeric: 0 - không, 1 – tiểu học (4th grade),  2 - THCS 5th đến 9th grade, 3 -  THPT và 4 – Đại học)
```bash
python program.py -p 12,13,0,0,0,4
```
* Chọn model để tìm điểm bằng tag -alg 
* Random forest là rf, linear là lr, mặc định là linear model
```bash
python program.py -p 12,13,0,0,0,4 -alg rf 
```
* Tính giá trị mae và rmse của 2 model
```bash
python program.py -m
```
* In ra hàm hồi quy tìm được của linear model
```bash
python program.py -f
```