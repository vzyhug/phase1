"""Build a self-contained Colab notebook for the three source-backed U4 blocks."""

import ast
import hashlib
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DRAW_SOURCE = ROOT / 'scripts/build_paper_figures.py'
CHECK_SOURCE = ROOT / 'scripts/fig02_architecture_diagram.py'
MODEL_NOTEBOOK = ROOT / 'notebooks/kaggle_ablation_u4_all_noise_512_100ep.ipynb'
DESTINATION = ROOT / 'notebooks/colab_draw_fig03_blocks.ipynb'
BLOCK_FILES = {
    'HNFBlock': 'src/models/blocks/hnf_block.py',
    'BridgeBlock': 'src/models/blocks/bridge_block.py',
    'SelfAttention1D': 'src/models/blocks/attention_1d.py',
}


def main():
    model_source = nbf.read(MODEL_NOTEBOOK, as_version=4).cells[6].source
    classes = {node.name: node for node in ast.parse(model_source).body
               if isinstance(node, ast.ClassDef)}
    hashes = {}
    for name, relative in BLOCK_FILES.items():
        source = (ROOT / relative).read_text(encoding='utf-8')
        original = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.ClassDef) and node.name == name)
        assert ast.dump(original) == ast.dump(classes[name]), f'{name} differs from embedded U4 code'
        hashes[name] = hashlib.sha256(ast.dump(original).encode('utf-8')).hexdigest()

    source = DRAW_SOURCE.read_text(encoding='utf-8')
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    helpers = '\n\n\n'.join(functions[name] for name in
                              ('canvas', 'label', 'box', 'circle', 'arrow', 'line'))
    helpers = helpers.replace('fontsize=size, color=color, fontweight=weight,',
                              'fontsize=size * FONT_SCALE, color=color, fontweight=weight,')
    checks = CHECK_SOURCE.read_text(encoding='utf-8')
    checker = next(ast.get_source_segment(checks, node) for node in ast.parse(checks).body
                   if isinstance(node, ast.FunctionDef) and node.name == 'check_diagram')
    helpers += '\n\n\n' + checker

    body = textwrap.dedent('\n'.join(functions['blocks_figure'].splitlines()[1:]))
    a = body.index('h = box(')
    b = body.index('s = box(')
    c = body.index('e4 = box(')
    end = body.index('save(fig, "fig03_blocks")')
    parts = (body[a:b], body[b:c], body[c:end])
    titles = [next(line for line in body.splitlines() if f'({key})' in line and '8.08' in line)
              for key in 'abc']
    panels = ['def draw_panel_' + key + '(ax):\n'
              + textwrap.indent(title + '\n' + part.strip(), '    ')
              for key, title, part in zip('abc', titles, parts)]
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

    cells = [
        md(r'''# Hình 3 — Chi tiết các khối HNF, FiLM và self-attention

Notebook Colab **vẽ sơ đồ phương pháp**, không huấn luyện hoặc đánh giá mô hình. Mã vẽ được nhúng đầy đủ; không cần GPU, dataset, checkpoint, ảnh gốc hoặc thư mục `src`.

**Cách dùng:** Colab CPU → **Run all**. Xem từng phần ở **6a, 6b, 6c**, bản ghép ở **mục 7**. Không tự tải file. Khi chốt, bật xuất tại **mục 8** để lấy `3.png`, `3.pdf`, `3.svg`; có thể xuất thêm `3a`, `3b`, `3c`.

- **(a) HNF:** bốn nhánh Conv1D đa tỉ lệ, half instance normalization và residual.
- **(b) FiLM:** embedding mức diffusion → Linear → SiLU → scale/shift theo kênh.
- **(c) Self-attention:** tạo Q/K/V, chia head, attention theo thời gian và chiếu đầu ra.

Mỗi phần có luồng chính từ trên xuống. Bản ghép đặt ba phần cạnh nhau, khổ hai cột IEEE **7.16 × 4.8 inch**, không giảm cỡ chữ khi ghép. Cấu hình và phép tính được đối chiếu với **U4 mix 512**; các lớp trong `src` và notebook huấn luyện đã được kiểm tra khớp AST khi tạo notebook này.
'''),
        md(r'''## 1. Nội dung cần đọc đúng

**HNF:** mỗi nhánh kernel $3,5,9,15$ sinh $C_o/4$ kênh, dùng reflection padding để giữ chiều dài. Sau nối là Conv1D kernel 1, chia đôi kênh, chỉ InstanceNorm nửa đầu, nối lại rồi ReLU. **Residual cộng sau ReLU**; nhánh tắt là Identity hoặc Conv1D kernel 1 khi đổi số kênh.

**FiLM:** $s=\sqrt{\bar\alpha}$ → embedding hình sin 128 chiều → Linear $128\to2C$ → SiLU → chia thành $\gamma(s),\xi(s)$. Công thức:

\[
h'=h\odot(1+\gamma(s))+\xi(s).
\]

Scale/shift theo kênh và broadcast theo chiều thời gian. FiLM nằm tại bốn encoder, không nằm tại decoder hoặc trên nhánh skip.

**Self-attention:** ví dụ bottleneck U4 có $640\times64$, chia 4 head nên $d_h=160$. Trong công thức hiển thị, mỗi $Q_h,K_h,V_h$ được hiểu theo dạng $L_b\times d_h$:

\[
A_h=\operatorname{softmax}(Q_hK_h^T/\sqrt{d_h}),\qquad O_h=A_hV_h.
\]

Nối 4 head và Conv1D kernel 1 để chiếu đầu ra. Không thêm lớp norm, dropout hoặc residual bao ngoài vì mã nguồn hiện hành không có. Kích thước 640 × 64 là kích thước kiến trúc cho đoạn 512, không phải kết quả đo.
'''),
        md('## 2. Cấu hình hình\n\nCó thể chỉnh màu, cỡ chữ, độ nét và khổ hình. `FONT_SCALE` dùng chung cho bản riêng/bản ghép; không tự thu nhỏ chữ. `GRAYSCALE=True` xem bản xám. Khi thay chữ khiến vượt khung, hãy tăng khung/ngắt dòng. Chế độ mặc định không tạo thư mục hoặc tải file.'),
        code('''from pathlib import Path
from io import BytesIO
import json
import zipfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
from IPython.display import display, Image

COMBINED_WIDTH_IN, COMBINED_HEIGHT_IN = 7.16, 4.8
PANEL_WIDTH_IN, PANEL_HEIGHT_IN = 3.5, 5.2
FONT_SCALE = 1.0
PREVIEW_DPI, PNG_DPI = 160, 600
GRAYSCALE = False
STRICT_TEXT_CHECK = True
PANEL_BOUNDS = {'a': (0, 5.35), 'b': (5.35, 10.45), 'c': (10.45, 16)}
INK, MUTED = '#233240', '#526171'
BLUE, GOLD = '#245A81', '#8C651F'
HNF_FILL, FILM_FILL, ATTN_FILL = '#E9F1F7', '#FBF2DD', '#F0EBF5'
GRAY_FILL, WHITE = '#F4F6F8', '#FFFFFF'
if GRAYSCALE:
    BLUE = GOLD = INK
    HNF_FILL, FILM_FILL, ATTN_FILL = '#E7E7E7', '#F4F4F4', '#DADADA'
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 8.2, 'mathtext.fontset': 'dejavuserif',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'svg.hashsalt': 'ecg-u4-fig03', 'savefig.facecolor': WHITE, 'axes.unicode_minus': False,
})
OUTPUT_DIR = (Path('/content') if Path('/content').exists() else Path.cwd()) / 'fig03_blocks'
SOURCE_BLOCK_SHA256 = __HASHES__
print('Matplotlib:', matplotlib.__version__)
print('Chỉ xem hình trong notebook; chưa xuất hoặc tải file.')
'''.replace('__HASHES__', repr(hashes))),
        md('## 3. Hàm vẽ và kiểm tra bố cục'), code(helpers),
        md('## 4. Phần (a) — HNF\n\nSửa nhãn, khung hoặc đường nối của HNF trong cell dưới. Hình dùng ký hiệu tổng quát $C_i,C_o,L$ vì cùng khối được dùng ở nhiều mức encoder/decoder.'),
        code(panels[0]),
        md('## 5. Phần (b), (c) — FiLM và self-attention\n\nHai hàm được tách thành hai cell để chỉnh riêng. Ký hiệu shift dùng $\\xi$ để không nhầm với nhiễu Gaussian $\\epsilon$.'),
        code(panels[1]), code(panels[2]),
        md('## 6. Chuẩn bị xem riêng từng phần\n\nChạy cell định nghĩa dưới một lần. Sau khi sửa phần nào, chạy lại hàm vẽ của phần đó rồi chỉ chạy cell xem tương ứng 6a/6b/6c.'),
        code('''def make_blocks_figure(panel=None):
    draws = {'a': draw_panel_a, 'b': draw_panel_b, 'c': draw_panel_c}
    if panel is not None and panel not in draws:
        raise ValueError('panel phải là a, b, c hoặc None cho bản ghép.')
    width = COMBINED_WIDTH_IN if panel is None else PANEL_WIDTH_IN
    height = COMBINED_HEIGHT_IN if panel is None else PANEL_HEIGHT_IN
    fig, ax = canvas(width, height, 16, 8.35)
    try:
        if panel is None:
            for draw in draws.values():
                draw(ax)
            for x in (5.35, 10.45):
                line(ax, [(x, 0.18), (x, 8.0)], color='#CBD2D9', width=0.5)
        else:
            ax.set_xlim(*PANEL_BOUNDS[panel])
            draws[panel](ax)
        check_diagram(fig)
        return fig
    except Exception:
        plt.close(fig)
        raise

def preview_blocks(panel=None):
    fig = make_blocks_figure(panel)
    try:
        with BytesIO() as buffer:
            fig.savefig(buffer, format='png', dpi=PREVIEW_DPI)
            display(Image(data=buffer.getvalue()))
    finally:
        plt.close(fig)
'''),
        md('### 6a. Xem riêng HNF'), code("preview_blocks('a')"),
        md('### 6b. Xem riêng FiLM'), code("preview_blocks('b')"),
        md('### 6c. Xem riêng self-attention'), code("preview_blocks('c')"),
        md('## 7. Xem Hình 3 hoàn chỉnh\n\nBản ghép dùng cùng hàm vẽ với bản riêng, không cắt/dán ảnh raster. Xem ở khổ hai cột paper trước khi xuất.'),
        code('preview_blocks()'),
        md('## 8. Xuất và tải — tùy chọn\n\nMặc định Run all **không xuất hoặc tải file**. Đặt `EXPORT_FILES=True` rồi chạy riêng cell này khi đã chốt. `EXPORT_PANELS=True` xuất thêm ba phần riêng. Chỉ bật `DOWNLOAD_ZIP=True` nếu muốn tải ZIP về máy. Bản ghép có tên **3.png / 3.pdf / 3.svg**, các bản riêng là 3a/3b/3c.'),
        code('''EXPORT_FILES = False
DOWNLOAD_ZIP = False
EXPORT_PANELS = True

def export_blocks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for panel in ([None, 'a', 'b', 'c'] if EXPORT_PANELS else [None]):
        fig = make_blocks_figure(panel)
        try:
            basename = '3' if panel is None else f'3{panel}'
            for extension in ('pdf', 'svg', 'png'):
                target = OUTPUT_DIR / f'{basename}.{extension}'
                options = {}
                if extension == 'pdf':
                    options['metadata'] = {'Title': basename, 'CreationDate': None, 'ModDate': None}
                elif extension == 'svg':
                    options['metadata'] = {'Date': None}
                else:
                    options['dpi'] = PNG_DPI
                fig.savefig(target, **options)
                paths.append(target)
        finally:
            plt.close(fig)
    spec_path = OUTPUT_DIR / 'figure_specification.json'
    spec_path.write_text(json.dumps({
        'kind': 'method_diagram_not_experimental_result', 'figure': 3,
        'source_notebook': 'kaggle_ablation_u4_all_noise_512_100ep.ipynb',
        'source_block_sha256': SOURCE_BLOCK_SHA256,
        'panels': {'a': 'HNF', 'b': 'FiLM', 'c': 'self_attention'},
        'combined_inches': [COMBINED_WIDTH_IN, COMBINED_HEIGHT_IN],
        'separate_inches': [PANEL_WIDTH_IN, PANEL_HEIGHT_IN],
        'font_scale': FONT_SCALE, 'png_dpi': PNG_DPI, 'grayscale': GRAYSCALE,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    paths.append(spec_path)
    zip_path = OUTPUT_DIR / 'fig03_blocks.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    print('Đã xuất:', ', '.join(path.name for path in paths))
    print('ZIP:', zip_path)
    return zip_path

if EXPORT_FILES:
    zip_path = export_blocks()
    if DOWNLOAD_ZIP:
        try:
            from google.colab import files
        except ImportError:
            print('Ngoài Colab: mở ZIP theo đường dẫn trên.')
        else:
            files.download(str(zip_path))
else:
    print('Chỉ xem trong notebook; chưa xuất/tải file.')
'''),
        md(r'''## 9. Chèn vào Overleaf bằng tên file

Tải `3.png` lên thư mục gốc dự án Overleaf; dùng trực tiếp **`{3.png}`**, không ghi đường dẫn máy tính. PDF vector có thể dùng bằng cách đổi thành `{3.pdf}`. Đường dẫn thư mục chỉ dành cho nơi lưu bản cục bộ, không cần trong câu lệnh chèn ảnh Overleaf.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{3.png}
  \caption{Các khối thành phần: (a) HNF đa tỉ lệ, chuẩn hóa nửa số kênh
  và cộng residual sau ReLU; (b) FiLM điều chế affine theo mức khuếch tán;
  (c) self-attention bốn head tại bottleneck, không có residual bao ngoài.}
  \label{fig:blocks}
\end{figure*}
```

Notebook không tự sửa paper hoặc thay ảnh Hình 1/2. Kiểm tra nhánh residual HNF, phép $1+\gamma$, vị trí SiLU và nhánh $V_h$ trước khi chốt. Khi thay cỡ chữ, chạy lại từ cấu hình để bản riêng và bản ghép thống nhất.
'''),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'name': DESTINATION.name, 'provenance': []},
    })
    nbf.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'{DESTINATION.name}:cell{index}')
    nbf.write(notebook, DESTINATION)
    print(f'{DESTINATION}\nValidated {len(cells)} cells; HNF/FiLM/attention match U4 source.')


if __name__ == '__main__':
    main()
