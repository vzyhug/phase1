"""Build a self-contained Colab drawing notebook for system Figure 1 only.

The existing, reviewed diagram code is embedded in notebook cells at build time.
The exported notebook never reads the repository, an existing image, ECG data,
or a checkpoint. Building/validating it does not render or export any figure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DRAW_SOURCE = ROOT / "scripts" / "build_paper_figures.py"
PANEL_SOURCE = ROOT / "scripts" / "fig01_vertical_panels.py"
DESTINATION = ROOT / "notebooks" / "colab_draw_fig01_system.ipynb"


def main():
    source = DRAW_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in tree.body if isinstance(node, ast.FunctionDef)}

    helpers = "\n\n\n".join(functions[name] for name in
                                ("canvas", "label", "box", "arrow", "line", "panel_title"))
    helpers = helpers.replace("fontsize=size, color=color, fontweight=weight,",
                              "fontsize=size * FONT_SCALE, color=color, fontweight=weight,")
    helpers += '''


def check_text_fit(fig):
    """Report labels too large for their boxes; never silently reduce type."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for patch, artist in getattr(fig, '_diagram_box_labels', []):
        a = patch.get_window_extent(renderer)
        b = artist.get_window_extent(renderer)
        if b.x0 < a.x0 or b.x1 > a.x1 or b.y0 < a.y0 or b.y1 > a.y1:
            overflow.append(artist.get_text().replace('\\n', ' / '))
    if overflow:
        print('Lưu ý: kiểm tra các nhãn có thể vượt khung:')
        for item in overflow:
            print(' -', item)
        if STRICT_TEXT_CHECK:
            raise ValueError('Nhãn vượt khung: tăng kích thước hình hoặc ngắt dòng nhãn.')
    else:
        print('Kiểm tra nhãn: chữ nằm trong các khối.')
    return overflow
'''

    panel_source = PANEL_SOURCE.read_text(encoding="utf-8")
    panel_functions = {node.name: ast.get_source_segment(panel_source, node)
                       for node in ast.parse(panel_source).body
                       if isinstance(node, ast.FunctionDef)}
    panel_code = [panel_functions[f"draw_panel_{name}"] for name in "abc"]

    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md(r"""# Vẽ Hình 1: Sơ đồ hệ thống khử nhiễu ECG — Google Colab

Notebook tạo **ba sơ đồ dọc độc lập (a), (b), (c)** theo thiết kế nghiên cứu 512 điểm. Luồng chính đi **từ trên xuống**, không xoay chữ hoặc thu nhỏ sơ đồ ngang:

- **(a)** Chuẩn bị ECG tham chiếu từ QTDB và nhiễu từ NSTDB.
- **(b)** Huấn luyện conditional diffusion với HNF–FiLM 1D U-Net.
- **(c)** Khôi phục ECG bằng quá trình đảo diffusion có điều kiện (DDIM).

Đây là notebook **vẽ sơ đồ phương pháp**, không thực hiện huấn luyện hoặc suy luận mô hình. Không cần GPU, dataset, checkpoint, ảnh gốc hay thư mục `src`. Mã vẽ đã được nhúng đầy đủ trong các cell.

**Cách dùng:** mở `.ipynb` trên Colab → chọn runtime CPU thông thường → **Runtime → Run all**. Ba hình a/b/c hiện ở **ba cell riêng tại mục 9a, 9b, 9c**, không tự ghép, xuất file hay tải xuống. Chỉnh cell `draw_panel_a/b/c`, chạy lại cell vừa sửa rồi chạy riêng cell xem trước tương ứng. Chỉ bật ghép hình ở mục 9d hoặc xuất/tải ở mục 10 khi đã hài lòng.

**Khổ chèn paper:** mỗi hình rộng mặc định **3.5 inch**, gần một cột IEEE, chữ nội dung khoảng 8.2 pt ở kích thước xuất gốc. Dùng từng PDF riêng với `width=\columnwidth`, không ghép cả ba vào một cột. Các nhánh điều kiện/vòng lặp đi bên cạnh nhưng luồng chính luôn đi xuống.

**Trạng thái bàn giao:** đã kiểm tra cấu trúc notebook và cú pháp Python; chưa thực thi các cell vẽ/xuất hình trên Colab. Các cell không có kết quả thực nghiệm được điền sẵn.
"""),
        md(r"""## 1. Nội dung phần (a): Chuẩn bị dữ liệu

**Luồng ECG:** QTDB → chọn ECG một kênh → tách đoạn theo nhịp → resample 360 Hz → đệm thành tensor dài 512 điểm.

**Luồng nhiễu:** NSTDB cung cấp BW (trôi đường nền), EM (chuyển động điện cực), MA (nhiễu cơ). Mỗi đoạn được gán một trong bảy tổ hợp không rỗng: BW, EM, MA, BW+EM, BW+MA, EM+MA, BW+EM+MA.

Các patch được cộng trước rồi hỗn hợp được scale một lần:

\[
n_{\mathrm{mix}}=m_1 n_{BW}+m_2 n_{EM}+m_3 n_{MA},\qquad
y=x_0+\kappa n_{\mathrm{mix}}.
\]

Đầu ra là cặp ECG tham chiếu–nhiễu $(x_0,y)$ có cùng kích thước. Hai tín hiệu dùng chung hệ số chuẩn hóa. Mask, cường độ, mã đoạn và split được cố định/lưu lại để các mô hình dùng cùng cặp dữ liệu. Việc chuẩn hóa hiện tại trong giao thức train 512 dùng scale từ ECG tham chiếu; sơ đồ không thay đổi hoặc tự suy ra một quy tắc triển khai mới.

**Bố cục dọc:** QTDB và NSTDB ở đầu hai nhánh song song, cùng đi xuống khâu tạo cặp → chuẩn hóa → cố định giao thức → cặp dữ liệu đầu ra.
"""),
        md(r"""## 2. Nội dung phần (b): Huấn luyện có điều kiện

ECG tham chiếu $x_0$ được thêm Gaussian $\epsilon\sim\mathcal{N}(0,I)$ ở mức liên tục $s$ để tạo trạng thái diffusion:

\[
x_s=sx_0+\sqrt{1-s^2}\,\epsilon.
\]

Mạng đầy đủ U4 (HNF, FiLM và attention ở bottleneck) nhận $(x_s,y,s)$ và dự đoán $\hat\epsilon$. ECG nhiễu $y$ được nối với trạng thái diffusion ở đầu vào; $s$ được nhúng và đưa qua FiLM tại các mức encoder.

\[
\hat\epsilon=\epsilon_\theta(x_s,y,s),\qquad
\hat x_0=\frac{x_s-\sqrt{1-s^2}\hat\epsilon}{s}.
\]

Loss kết hợp gồm:

- $\mathcal L_\epsilon$: L1 trung bình giữa Gaussian thật $\epsilon$ và dự đoán $\hat\epsilon$.
- $\mathcal L_f$: sai số tương đối biên độ STFT giữa $x_0$ và $\hat x_0$, có trọng số diffusion SNR.
- $\mathcal L=\mathcal L_\epsilon+0.1\mathcal L_f$.

Mũi tên nét đứt quay từ loss về mạng biểu diễn lan truyền ngược và cập nhật Adam. Checkpoint được chọn theo validation loss của mạng hiện hành; trọng số mạng và EMA tại epoch tương ứng được lưu cùng nhau. Không dùng metric test để chọn epoch.

**Bố cục dọc:** cặp huấn luyện/Gaussian/mức $s$ → diffusion thuận → mạng → dự đoán Gaussian → ước lượng $\hat x_0$ → loss → cập nhật Adam → lưu checkpoint. Nhánh $(y,s)$ và nhánh $\hat\epsilon$ đi bên phải; đường cập nhật trọng số quay về mạng ở bên trái. Tái tạo $\hat x_0$ chỉ phục vụ loss phổ; loss dự đoán Gaussian vẫn nhận trực tiếp $\hat\epsilon$.
"""),
        md(r"""## 3. Nội dung phần (c): Suy luận bằng DDIM

Sau khi đã huấn luyện, nạp checkpoint được chọn. Đầu vào của quá trình khôi phục là ECG quan sát $y$ trong miền chuẩn hóa và trạng thái khởi tạo Gaussian $x_T\sim\mathcal N(0,I)$.

Ở mỗi mốc $\tau$, cùng một mạng dự đoán:

\[
\hat\epsilon_\tau=\epsilon_\theta(x_\tau,y,s_\tau),
\]

sau đó DDIM cập nhật trạng thái về mốc tiếp theo cho đến khi thu được $\hat x_0\in\mathbb R^{1\times512}$. ECG $y$ được giữ làm điều kiện trong toàn bộ quá trình; không cần tạo lại nhiễu BW/EM/MA khi thực hiện vòng lặp DDIM.

**Bố cục dọc:** nạp checkpoint → đầu vào $y,x_T$ → dự đoán Gaussian → cập nhật DDIM → kiểm tra mốc tiếp theo → ECG phục hồi. Nhánh “Có” quay lên khối dự đoán ở cạnh phải; nhánh “Không” tiếp tục đi xuống đầu ra.

**Phân biệt hai nguồn nhiễu:** BW/EM/MA tạo tín hiệu quan sát $y$ ở (a); Gaussian $\epsilon$ phục vụ diffusion ở (b)/(c). Hình này mô tả thiết kế, không khẳng định checkpoint hỗn hợp 512 đã được huấn luyện hay đã đạt kết quả nào.
"""),
        md("## 4. Cấu hình hiển thị dọc\n\n`PANEL_WIDTH_IN=3.5` là chiều rộng xuất của từng hình; `PANEL_HEIGHTS_IN` điều chỉnh chiều cao riêng a/b/c. Ba hình luôn được xem riêng ở mục 9a/9b/9c. `SHOW_COMBINED_PREVIEW=False` tắt bản ghép tổng quan; `EXPORT_COMBINED=False` ưu tiên xuất từng hình dọc cho paper. Muốn chỉnh nội dung hoặc vị trí khối, sửa các cell `draw_panel_a/b/c`. Sau khi sửa cấu hình, chạy lại từ cell này đến mục 9. Màu, `FONT_SCALE` và `PREVIEW_DPI` dùng chung để kiểm tra ngay trong Colab."),
        code('''from pathlib import Path
from io import BytesIO
import json
import zipfile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
from IPython.display import display, Image

OUTPUT_DIR = (Path('/content') if Path('/content').exists() else Path.cwd()) / 'ecg_fig01_system'
PANEL_WIDTH_IN = 3.5     # Khổ một cột IEEE; không nén sơ đồ ngang
PANEL_HEIGHTS_IN = {'a': 7.0, 'b': 7.7, 'c': 5.95}
PANEL_YMAX = {'a': 20.0, 'b': 22.0, 'c': 17.0}
FONT_SCALE = 1.0
PNG_DPI = 600
GRAYSCALE = False
EXPORT_PANELS = True    # Xuất thêm các phần a, b, c riêng
EXPORT_COMBINED = False # Bản ghép rộng chỉ dùng tổng quan, không chèn vào một cột
PREVIEW_DPI = 160       # Độ nét hình hiển thị ngay dưới cell, độc lập PNG_DPI
SHOW_COMBINED_PREVIEW = False  # Chỉ bật khi muốn xem bản ghép sau khi kiểm tra a/b/c
STRICT_TEXT_CHECK = False

INK, MUTED = '#233240', '#526171'
BLUE, GOLD = '#245A81', '#8C651F'
HNF_FILL, FILM_FILL = '#E9F1F7', '#FBF2DD'
GRAY_FILL, WHITE = '#F4F6F8', '#FFFFFF'
if GRAYSCALE:
    BLUE = GOLD = INK
    HNF_FILL, FILM_FILL, GRAY_FILL = '#E9E9E9', '#F3F3F3', '#F8F8F8'

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 8.2,
    'mathtext.fontset': 'dejavuserif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'svg.hashsalt': 'ecg-hnf-film-fig01-colab',
    'savefig.facecolor': WHITE,
    'axes.unicode_minus': False,
})
print('Matplotlib:', matplotlib.__version__)
print('Chế độ xem trong Colab: không tự xuất hoặc tải file.')
'''),
        md("## 5. Các hàm vẽ cơ bản\n\nKhối chữ nhật biểu diễn một công đoạn, mũi tên liền biểu diễn luồng tín hiệu và mũi tên nét đứt biểu diễn điều kiện/cập nhật tham số. Hàm kiểm tra nhãn sẽ cảnh báo nếu thay đổi chữ hoặc font làm chữ vượt khung."),
        code(helpers),
        md("## 6. Vẽ phần (a): Nguồn ECG, nguồn nhiễu và cặp dữ liệu"),
        code(panel_code[0]),
        md("## 7. Vẽ phần (b): Diffusion thuận, mạng dự đoán và loss"),
        code(panel_code[1]),
        md("## 8. Vẽ phần (c): Vòng lặp suy luận DDIM"),
        code(panel_code[2]),
        md("## 9. Xem từng hình dọc độc lập trong Colab\n\nChạy cell chuẩn bị dưới đây một lần, sau đó xem riêng **9a, 9b, 9c**. Mỗi cell chỉ vẽ phần tương ứng trên canvas dọc mới, không cắt hoặc xoay hình ngang. **Không ghi file, không tải xuống.** Ví dụ: sửa `draw_panel_b` → chạy lại cell định nghĩa đó → chạy riêng **9b**, không cần vẽ lại a/c."),
        code('''PANEL_BASENAMES = {
    'a': 'fig01a_data_preparation',
    'b': 'fig01b_training',
    'c': 'fig01c_inference',
}

def show_inline(fig):
    # Chỉ tạo ảnh trong RAM để Colab hiển thị, không ghi ra ổ đĩa.
    with BytesIO() as buffer:
        fig.savefig(buffer, format='png', dpi=PREVIEW_DPI)
        display(Image(data=buffer.getvalue()))

def make_panel_figure(name):
    draw = {'a': draw_panel_a, 'b': draw_panel_b, 'c': draw_panel_c}[name]
    panel_fig, panel_ax = canvas(PANEL_WIDTH_IN, PANEL_HEIGHTS_IN[name], 10, PANEL_YMAX[name])
    draw(panel_ax)
    return panel_fig, panel_ax

def preview_panel(name):
    # Chỉ gọi hàm vẽ của phần đang kiểm tra; không phụ thuộc hình khác.
    panel_fig, panel_ax = make_panel_figure(name)
    try:
        check_text_fit(panel_fig)
        show_inline(panel_fig)
    finally:
        plt.close(panel_fig)

def make_system_figure():
    # Bản tổng quan: ba panel dọc cạnh nhau, giữ nguyên khổ chữ, không nén.
    height = max(PANEL_HEIGHTS_IN.values())
    combined_fig = plt.figure(figsize=(3 * PANEL_WIDTH_IN, height), facecolor=WHITE)
    for index, name in enumerate('abc'):
        panel_height = PANEL_HEIGHTS_IN[name] / height
        axis = combined_fig.add_axes((
            (index + 0.008) / 3, 1 - 0.012 * panel_height - 0.976 * panel_height,
            0.984 / 3, 0.976 * panel_height,
        ))
        axis.set(xlim=(0, 10), ylim=(0, PANEL_YMAX[name]))
        axis.axis('off')
        {'a': draw_panel_a, 'b': draw_panel_b, 'c': draw_panel_c}[name](axis)
    return combined_fig
'''),
        md("### 9a. Hình (a) — Chuẩn bị dữ liệu\n\nKiểm tra luồng QTDB, NSTDB, tổ hợp nhiễu và cặp dữ liệu đầu ra. Chạy lại riêng cell này sau khi sửa `draw_panel_a`."),
        code("preview_panel('a')"),
        md("### 9b. Hình (b) — Huấn luyện conditional diffusion\n\nKiểm tra diffusion thuận, điều kiện đầu vào, mạng dự đoán và các nhánh loss. Chạy lại riêng cell này sau khi sửa `draw_panel_b`."),
        code("preview_panel('b')"),
        md("### 9c. Hình (c) — Suy luận DDIM\n\nKiểm tra đầu vào ECG nhiễu, khởi tạo Gaussian, vòng lặp DDIM và ECG phục hồi. Chạy lại riêng cell này sau khi sửa `draw_panel_c`."),
        code("preview_panel('c')"),
        md("### 9d. Bản tổng quan ba hình dọc — tùy chọn, mặc định tắt\n\nSau khi kiểm tra ba hình riêng, đặt `SHOW_COMBINED_PREVIEW=True` ở mục 4, chạy lại cell cấu hình rồi cell dưới đây. Bản tổng quan đặt ba panel dọc cạnh nhau ở khổ rộng **10.5 inch**, không thu nhỏ chữ. Khi chèn vào paper, ưu tiên ba PDF riêng thay vì ép bản rộng này vào một cột."),
        code('''if SHOW_COMBINED_PREVIEW:
    combined_fig = make_system_figure()
    try:
        check_text_fit(combined_fig)
        show_inline(combined_fig)
    finally:
        plt.close(combined_fig)
else:
    print('Đang kiểm tra riêng a/b/c; hình ghép đã tắt.')
'''),
        md("## 10. Xuất file và tải xuống — tùy chọn\n\nMặc định cell này **không làm gì ngoài thông báo** khi Run all. Khi đã chỉnh xong, đặt `EXPORT_FILES=True` rồi chạy riêng cell này để tạo PDF/SVG/PNG **riêng cho từng hình dọc** và ZIP trong thư mục Files của Colab. Chỉ đặt thêm `DOWNLOAD_ZIP=True` nếu muốn tải về máy. Bước xuất tự vẽ lại từ các hàm hiện hành. `EXPORT_COMBINED=True` xuất thêm bản tổng quan `fig01_system.*` nếu cần, mặc định không xuất bản rộng này. PDF/SVG là vector; ưu tiên dùng từng PDF với `width=\\columnwidth` trong paper."),
        code('''EXPORT_FILES = False   # Chỉ đổi True khi đã xem/chỉnh xong hình
DOWNLOAD_ZIP = False   # Chỉ đổi True khi thực sự muốn tải về máy

def export_figure(fig, basename):
    saved = []
    for extension in ('pdf', 'svg', 'png'):
        path = OUTPUT_DIR / f'{basename}.{extension}'
        options = {}
        if extension == 'pdf':
            options['metadata'] = {'Title': basename, 'CreationDate': None, 'ModDate': None}
        elif extension == 'svg':
            options['metadata'] = {'Date': None}
        else:
            options['dpi'] = PNG_DPI
        fig.savefig(path, **options)
        saved.append(path)
    return saved

def export_bundle():
    if not EXPORT_PANELS and not EXPORT_COMBINED:
        raise ValueError('Bật EXPORT_PANELS hoặc EXPORT_COMBINED trước khi xuất.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_paths, text_warnings = [], {}
    if EXPORT_PANELS:
        for name, basename in PANEL_BASENAMES.items():
            fig, ax = make_panel_figure(name)
            try:
                text_warnings[name] = check_text_fit(fig)
                export_paths.extend(export_figure(fig, basename))
            finally:
                plt.close(fig)
    if EXPORT_COMBINED:
        fig = make_system_figure()
        try:
            text_warnings['combined'] = check_text_fit(fig)
            export_paths.extend(export_figure(fig, 'fig01_system'))
        finally:
            plt.close(fig)
    specification = {
    'figure': 'fig01_system',
    'kind': 'method_diagram_not_experimental_result',
    'panels': {
        'a': 'QTDB + NSTDB; 7 noise subsets; paired data preparation',
        'b': 'conditional diffusion training; noise prediction; weighted spectral objective',
        'c': 'conditional DDIM inference from Gaussian initialization',
    },
    'ecg_channels': 1,
    'tensor_length': 512,
    'resample_hz': 360,
    'lambda_frequency': 0.1,
    'layout': 'portrait_panels_top_to_bottom',
    'panel_width_inches': PANEL_WIDTH_IN,
    'panel_heights_inches': PANEL_HEIGHTS_IN,
    'export_combined': EXPORT_COMBINED,
    'png_dpi': PNG_DPI,
    'font_scale': FONT_SCALE,
    'grayscale': GRAYSCALE,
    'text_fit_warnings': text_warnings,
    }
    spec_path = OUTPUT_DIR / 'figure_specification.json'
    spec_path.write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding='utf-8')
    export_paths.append(spec_path)
    zip_path = OUTPUT_DIR / 'fig01_system_colab.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in export_paths:
            archive.write(path, arcname=path.name)
    for path in export_paths:
        print(path.name, f'({path.stat().st_size:,} bytes)')
    print('ZIP:', zip_path)
    return zip_path

if EXPORT_FILES:
    zip_path = export_bundle()
    if DOWNLOAD_ZIP:
        try:
            from google.colab import files
        except ImportError:
            print('Ngoài Colab: mở ZIP tại đường dẫn phía trên.')
        else:
            files.download(str(zip_path))
else:
    print('Đang chỉ xem hình trong notebook; chưa xuất hoặc tải file.')
    print('Khi cần: đặt EXPORT_FILES=True; DOWNLOAD_ZIP=True chỉ nếu muốn tải.')
'''),
        md(r"""## 11. Chú thích hình gợi ý cho bài báo

**Hình 1.** Luồng hoạt động của hệ thống khử nhiễu ECG: (a) tạo cặp ECG tham chiếu–nhiễu từ QTDB và bảy tổ hợp BW/EM/MA của NSTDB; (b) huấn luyện mạng dự đoán Gaussian với loss thời gian–tần số; (c) phục hồi ECG qua các bước DDIM có điều kiện. Gaussian trong quá trình diffusion khác với artifact cộng vào ECG quan sát. Các tín hiệu trong nhánh mô hình được biểu diễn ở miền đã chuẩn hóa.

```latex
% Chèn từng sơ đồ dọc vào một cột, không nén cả ba sơ đồ vào một cột.
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{fig01a_data_preparation.pdf}
  \caption{Chuẩn bị cặp dữ liệu ECG tham chiếu–nhiễu.}
  \label{fig:data-preparation}
\end{figure}
% Tương tự: fig01b_training.pdf và fig01c_inference.pdf.
```

Khi chỉnh sửa, kiểm tra chữ/đường nối ở kích thước in thực tế. Nếu cảnh báo nhãn xuất hiện, tăng chiều rộng/chiều cao hoặc ngắt dòng nội dung; không cần tải lại dữ liệu hay huấn luyện mô hình để vẽ sơ đồ này.
"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": DESTINATION.name, "provenance": []},
    })
    nbf.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{DESTINATION.name}:cell{index}")
    nbf.write(notebook, DESTINATION)
    print(DESTINATION)
    print(f"Validated {len(cells)} cells; drawing/export cells were not executed.")


if __name__ == "__main__":
    main()
