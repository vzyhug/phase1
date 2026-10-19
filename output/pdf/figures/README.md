# Hình phương pháp cho bài báo

| Tệp | Nội dung | Vị trí trong paper.tex |
|---|---|---|
| `fig01_system` | Chuẩn bị dữ liệu, huấn luyện và suy luận conditional diffusion | Hình 1, `fig:system` |
| `fig02_architecture` | U4: encoder, FiLM, skip, attention, decoder và kích thước tensor | Hình 2, `fig:architecture` |
| `fig03_blocks` | Chi tiết HNF, FiLM và self-attention | Hình 3, `fig:blocks` |

Mỗi hình có PDF vector để chèn vào LaTeX, SVG chứa chữ có thể chỉnh sửa và PNG 300 dpi để xem nhanh hoặc dùng trong báo cáo. Các sơ đồ mô tả phương pháp theo mã nguồn và cấu hình 512 điểm; chúng không phải kết quả thực nghiệm.

## Dùng trên Overleaf

Import `output/paper_overleaf.zip` thành một dự án, chọn `paper.tex` làm main document và dùng pdfLaTeX. ZIP đã chứa đường dẫn các PDF mà `paper.tex` sử dụng. Nếu thêm tệp thủ công, giữ nguyên cấu trúc `output/pdf/figures/` so với `paper.tex`.

## Tạo lại hoặc chỉnh sửa hình

Chỉnh `scripts/build_paper_figures.py`, sau đó chạy từ thư mục gốc:

```powershell
python scripts/build_paper_figures.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf paper.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf paper.tex
```

Script kiểm tra chữ nằm trong các khối trước khi xuất và ghi hash nguồn vào `figure_sources.json`. SVG có thể chỉnh bằng Inkscape hoặc phần mềm vector; nếu sửa SVG thủ công, cần xuất lại PDF tương ứng trước khi biên dịch bài.

Hình thuật toán huấn luyện hiện có được LaTeX đánh số thành Hình 4. Hình dạng sóng đang chờ prediction của cùng tập test (Hình 5); đồ thị hội tụ chờ log U0--U4 và đồ thị chất lượng--latency chờ kết quả khảo sát sampler.
