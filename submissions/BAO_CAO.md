# Báo Cáo Lab MLOps - CI/CD cho AI Systems

**Sinh viên:** Nguyễn Thị Vang
**Mã số:** 2A202600723
**Repo:** https://github.com/WinestrytoCode/Day21-2A202600723-NguyenThiVang-Track2-CI-CD-for-AI-Systems

---

## 1. Bộ Siêu Tham Số Đã Chọn Và Lý Do

### Kết quả thí nghiệm (Bước 1 - MLflow tracking)

Bài toán: phân loại chất lượng rượu vang thành 3 lớp (thấp / trung bình / cao). Em đã chạy nhiều thí nghiệm với các thuật toán và siêu tham số khác nhau trên tập `train_phase1` (2998 mẫu), đánh giá trên `eval.csv` (500 mẫu held-out):

| Thuật toán | Siêu tham số | Accuracy | F1-score |
|---|---|---|---|
| RandomForest | n_estimators=10, max_depth=3 | 0.275 | 0.279 |
| RandomForest | n_estimators=50, max_depth=3 | 0.558 | 0.518 |
| RandomForest | n_estimators=100, max_depth=5 | 0.564 | 0.553 |
| RandomForest | n_estimators=200, max_depth=10 | 0.620 | 0.618 |
| **ExtraTrees** | **n_estimators=300, min_samples_split=2** | **0.684** | **0.681** |

### Bộ tham số được chọn

```yaml
model_type: extra_trees
n_estimators: 300
min_samples_split: 2
```

**Lý do chọn:**

- **Số cây lớn (n_estimators=300):** Tăng số cây giúp giảm phương sai (variance) và ổn định kết quả. Khi tăng từ 10 → 50 → 100 → 200 cây, accuracy tăng đều (0.275 → 0.558 → 0.564 → 0.620), cho thấy mô hình ban đầu bị underfit do quá ít cây.

- **Không giới hạn max_depth (để cây phát triển đầy đủ):** Việc giới hạn độ sâu (max_depth=3, 5) làm cây quá đơn giản, không nắm được quan hệ phi tuyến giữa các đặc trưng hóa học và chất lượng rượu. Bỏ giới hạn độ sâu cho accuracy cao hơn rõ rệt.

- **min_samples_split=2:** Cho phép cây tách nhánh tối đa, khai thác hết thông tin từ dữ liệu — phù hợp với mô hình ensemble nhiều cây vốn đã chống overfit tốt.

- **ExtraTrees thay vì RandomForest:** ExtraTrees (Extremely Randomized Trees) chọn điểm cắt ngẫu nhiên thay vì tìm điểm cắt tối ưu như RandomForest. Điều này tăng tính ngẫu nhiên, giảm overfit, và trong thí nghiệm cho accuracy cao nhất (0.684 so với 0.620 của RandomForest tốt nhất).

---

## 2. Khó Khăn Gặp Phải Và Cách Giải Quyết

### 2.1. Lỗi cài đặt thư viện trên Python 3.13

**Vấn đề:** `pip install -r requirements.txt` thất bại vì các phiên bản pin trong file gốc (scikit-learn 1.4.2, pandas 2.2.2, mlflow 2.13.0) không có wheel cho Python 3.13, phải build từ source và lỗi (`numpy==2.0.0rc1 not found`, `pkg_resources missing`).

**Giải pháp:** Nâng cấp các phiên bản tương thích Python 3.13: scikit-learn → 1.5.2, pandas → 2.2.3, mlflow → 2.22.5, pyyaml → 6.0.2.

### 2.2. Unit Test thất bại trên GitHub Actions (nhưng pass ở local)

**Vấn đề:** Test crash với `MissingConfigException: mlruns/0/meta.yaml does not exist`. Thư mục `mlruns/` bị commit vào git nhưng thiếu file metadata khi checkout trên CI runner sạch.

**Giải pháp:** Thêm pytest fixture (autouse) trỏ MLflow vào thư mục tạm riêng cho mỗi test, và thêm `mlruns/` vào `.gitignore` để không commit thư mục output.

### 2.3. Lỗi xác thực GCS trong CI - "no remote" rồi "Invalid Credentials 401"

**Vấn đề:** DVC pull thất bại nhiều lần: ban đầu vì `.dvc/config` trống (chưa cấu hình remote), sau đó vì `credentialpath` trong config trỏ tới file chỉ tồn tại ở local nên CI bị 401.

**Giải pháp:** Cấu hình DVC remote trỏ tới `gs://mlops-lab-vang-2026/dvc`, gỡ `credentialpath` khỏi config để DVC dùng biến môi trường `GOOGLE_APPLICATION_CREDENTIALS` (hoạt động cho cả local lẫn CI).

### 2.4. "Anonymous caller" - service account key bị hỏng trong CI

**Vấn đề:** DVC vẫn truy cập GCS dưới danh nghĩa "Anonymous" dù đã có bước Authenticate. Nguyên nhân: ghi secret bằng `echo` làm hỏng JSON (ký tự `\n` trong `private_key`), và thực tế repo chưa được thêm secret nào.

**Giải pháp:** Truyền secret qua biến môi trường rồi ghi bằng `printf '%s'` (giữ nguyên JSON), thêm bước verify `json.load`. Thêm đầy đủ các Repository Secrets trên GitHub.

### 2.5. Accuracy không đạt ngưỡng eval gate 0.70

**Vấn đề:** Với 2998 mẫu (phase1), accuracy tối đa chỉ ~0.68-0.69 dù tune kỹ — eval gate (>= 0.70) luôn chặn deploy.

**Giải pháp:** Thực hiện Bước 3 - bổ sung `train_phase2` nâng dữ liệu lên 5996 mẫu. Accuracy tăng lên **0.768**, vượt ngưỡng. Điều này chứng minh giá trị của continuous training: thêm dữ liệu cải thiện chất lượng mô hình.

### 2.6. Deploy thất bại - thiếu thư viện và timing

**Vấn đề:** Service trên VM crash (`ModuleNotFoundError: fastapi`) vì systemd dùng `/usr/bin/python3` không có thư viện. Sau khi sửa, deploy vẫn fail vì health check chỉ chờ 5 giây nhưng server cần ~8 giây để tải model từ GCS.

**Giải pháp:** Tạo virtualenv trên VM và trỏ `ExecStart` vào python của venv. Sửa workflow: thay `sleep 5 + curl 1 lần` bằng vòng lặp retry health check tối đa 12 lần (60s).

---

## 3. Tính Năng Nâng Cao Đã Triển Khai (Bonus)

- **Bonus 1 - Remote tracking:** Kết nối MLflow tới DagsHub qua biến môi trường.
- **Bonus 2 - Multi-algorithm:** Tham số `model_type` chọn RandomForest / GradientBoosting / ExtraTrees / LogisticRegression.
- **Bonus 3 - Báo cáo tự động:** Confusion matrix + precision/recall từng lớp ghi ra `outputs/report.txt`.
- **Bonus 4 - Rollback guard:** So sánh accuracy với model đang chạy, chặn deploy nếu model mới kém hơn.
- **Bonus 5 - Data drift:** Cảnh báo khi một lớp chiếm < 10% và ghi phân phối nhãn vào metrics.json.
