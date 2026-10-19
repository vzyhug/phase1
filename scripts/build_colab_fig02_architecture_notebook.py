"""Build the self-contained, preview-first Colab notebook for Figure 2."""

import ast
import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
MODEL_NOTEBOOK = ROOT / 'notebooks/kaggle_ablation_u4_all_noise_512_100ep.ipynb'
HELPER_SOURCE = ROOT / 'scripts/build_paper_figures.py'
DIAGRAM_SOURCE = ROOT / 'scripts/fig02_u_layout.py'
CHECK_SOURCE = ROOT / 'scripts/fig02_architecture_diagram.py'
DESTINATION = ROOT / 'notebooks/colab_draw_fig02_architecture.ipynb'


def main():
    model_notebook = nbf.read(MODEL_NOTEBOOK, as_version=4)
    architecture = model_notebook.cells[6].source
    model_tree = ast.parse(architecture)
    unet = next(node for node in model_tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'UNet1D')
    init = next(node for node in unet.body
                if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    assert [ast.literal_eval(node) for node in init.args.defaults] == [2, 80, 128, 1]
    assert 'self.attn = SelfAttention1D(c * 8)' in architecture
    assert 'self.bridge4(self.enc4(self.down3(e3)), noise_scale)' in architecture
    assert 'torch.cat([self.up4(e4), e3], dim=1)' in architecture
    assert 'torch.cat([self.up3(d4), e2], dim=1)' in architecture
    assert 'torch.cat([self.up2(d3), e1], dim=1)' in architecture
    configuration = model_notebook.cells[4].source
    assert 'LENGTH = 512' in configuration and "'feats': 80" in configuration
    assert "'id': 'U4'" in configuration
    assert 'def __init__(self, channels, num_heads=4)' in architecture
    source_hash = hashlib.sha256(architecture.encode('utf-8')).hexdigest()

    helper_source = HELPER_SOURCE.read_text(encoding='utf-8')
    functions = {node.name: ast.get_source_segment(helper_source, node)
                 for node in ast.parse(helper_source).body
                 if isinstance(node, ast.FunctionDef)}
    helpers = '\n\n\n'.join(functions[name] for name in
                              ('canvas', 'label', 'box', 'arrow', 'line'))
    helpers = helpers.replace('fontsize=size, color=color, fontweight=weight,',
                              'fontsize=size * FONT_SCALE, color=color, fontweight=weight,')
    diagram_source = DIAGRAM_SOURCE.read_text(encoding='utf-8')
    drawing = '\n\n\n'.join(ast.get_source_segment(diagram_source, node)
                              for node in ast.parse(diagram_source).body
                              if isinstance(node, ast.FunctionDef))
    checks = CHECK_SOURCE.read_text(encoding='utf-8')
    checker = next(ast.get_source_segment(checks, node) for node in ast.parse(checks).body
                   if isinstance(node, ast.FunctionDef) and node.name == 'check_diagram')
    drawing = checker + '\n\n\n' + drawing

    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    config = '''from pathlib import Path
from io import BytesIO
import json
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from IPython.display import display, Image

# Chỉ vẽ sơ đồ: không cần GPU, checkpoint, dataset hoặc thư mục src.
MODEL_SPEC = {
    'id': 'U4', 'length': 512, 'input_channels': 2, 'output_channels': 1,
    'base_channels': 80, 'embedding_dim': 128, 'attention_heads': 4,
}
SOURCE_NOTEBOOK = 'notebooks/kaggle_ablation_u4_all_noise_512_100ep.ipynb'
SOURCE_ARCHITECTURE_SHA256 = '__SOURCE_HASH__'

# Khổ hai cột IEEE. Giữ chữ ở cỡ in được, không ép sơ đồ này vào một cột.
FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 6.3
FONT_SCALE = 1.0
PREVIEW_DPI = 160
PNG_DPI = 600
GRAYSCALE = False
STRICT_TEXT_CHECK = True

# Tọa độ sơ đồ: x trong [0,22], y trong [0,20.3].
# Có thể chỉnh các tọa độ này rồi chạy lại phần vẽ/xem trước.
ENCODER_X = (4.0, 5.3, 6.6)
DECODER_X = (18.0, 16.7, 15.4)  # Cùng thứ tự mức 1, 2, 3.
LEVEL_Y = (12.4, 9.0, 5.6)
BOTTLENECK_Y = 2.1
STAGE_WIDTH, STAGE_HEIGHT = 4.8, 2.1

# Mặc định vẽ tín hiệu tổng hợp MINH HỌA, không phải kết quả huấn luyện.
# 'symbols': chỉ ký hiệu. 'provided': vẽ dữ liệu đã có từ mô hình.
SIGNAL_MODE = 'illustration'
ILLUSTRATION_SEED = 1234
SIGNALS_NPZ = None  # Với provided: .npz có y, x_s, epsilon_pred; mỗi vector dài 512.

INK, MUTED = '#233240', '#526171'
BLUE, GOLD = '#245A81', '#8C651F'
NOISE_COLOR = '#AE4A3A'
HNF_FILL, FILM_FILL = '#E9F1F7', '#FBF2DD'
DECODER_FILL = '#EFF2F5'
ATTN_FILL, GRAY_FILL, WHITE = '#F0EBF5', '#F4F6F8', '#FFFFFF'
if GRAYSCALE:
    BLUE = GOLD = INK
    NOISE_COLOR = INK
    HNF_FILL, FILM_FILL, ATTN_FILL = '#E7E7E7', '#F5F5F5', '#D9D9D9'
    DECODER_FILL = '#F0F0F0'

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 8.2,
    'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'svg.hashsalt': 'ecg-u4-fig02',
    'savefig.facecolor': WHITE, 'axes.unicode_minus': False,
})
OUTPUT_DIR = (Path('/content') if Path('/content').exists() else Path.cwd()) / 'fig02_architecture'
OUTPUT_BASENAME = '2'  # Xuất 2.png / 2.pdf / 2.svg để chèn paper.
print('Matplotlib:', matplotlib.__version__)
print('Chỉ xem trong notebook; chưa ghi hoặc tải file.')
'''.replace('__SOURCE_HASH__', source_hash)

    cells = [
        md(r'''# Hình 2 — Kiến trúc HNF–FiLM 1D U-Net (U4)

Notebook **vẽ kiến trúc**, không huấn luyện hoặc đánh giá mô hình. Mã vẽ được nhúng đầy đủ; **không cần GPU, dataset, checkpoint, file ảnh hoặc `src`**. Chỉ dùng Matplotlib và các thành phần có sẵn trong Colab.

**Cách dùng:** mở notebook trên Colab → runtime CPU → **Runtime → Run all**. Hình hiện ngay ở **mục 5**; không tự xuất/tải file. Sửa cấu hình hoặc hàm vẽ, chạy lại các cell liên quan rồi chạy lại mục 5. Xuất/tải là thao tác riêng ở mục 6, chỉ bật khi đã hài lòng.

**Bố cục chữ U:** ba encoder bên trái và ba decoder bên phải nằm đối xứng ở cùng độ phân giải, các nhánh càng xuống càng gần tâm. **Encoder 4 và attention được gom vào một khối bottleneck ở giữa đáy**, không đặt lệch sang một bên. Ba skip chạy ngang thẳng; không có đường vòng dài bao quanh hình.

Mỗi khối decoder gom chuỗi **Up → Concat → HNF**; đầu vào đặc trưng đi từ dưới lên, skip đi vào phép Concat bên trong khối, không đi vào Up. Nhãn trong khối ghi kích thước đầu ra HNF. Các tham chiếu ngắn $e(s)$ cùng chỉ embedding chung phía trên; đây là cách rút gọn đường dây điều kiện, không thêm hay bớt khối FiLM.

**Kiến trúc giữ đúng mã nguồn:** bốn encoder, ba decoder; FiLM tại bốn encoder, không đặt ở decoder hoặc trên đường skip. Đầu ra là Gaussian dự đoán, không phải ECG phục hồi trực tiếp.

**Waveform đầu vào/đầu ra:** bên trái là ECG nhiễu $y$ và trạng thái diffusion $x_s$; bên phải vẫn là Gaussian $\hat\epsilon$, không đổi thành ECG sạch. Mặc định các sóng là **tín hiệu tổng hợp minh họa**, được ghi rõ ngay trên hình; không dùng checkpoint, không phải dữ liệu QTDB/NSTDB hay kết quả đánh giá. Gaussian phía phải chỉ là ký họa cho loại tín hiệu đầu ra, không phải prediction đã tính từ mạng.

**Các chế độ tín hiệu:** `SIGNAL_MODE='illustration'` chạy ngay, không cần dữ liệu; `'symbols'` bỏ waveform; `'provided'` đọc file do bạn cung cấp qua `SIGNALS_NPZ` với các vector `y`, `x_s`, `epsilon_pred` của cùng một mẫu. Chế độ provided chỉ vẽ dữ liệu, không chạy mô hình. Không đưa ECG đã khử nhiễu vào `epsilon_pred`.

**Phân chia nội dung:** Hình 1 mô tả quy trình hệ thống; Hình 2 mô tả mạng với waveform giúp nhận diện loại tín hiệu; Hình 3 trình bày bên trong các khối. Waveform minh họa trong Hình 2 không dùng để kết luận chất lượng phục hồi.

**Khổ paper:** mặc định 7.16 × 6.3 inch, dành cho hình trải **hai cột IEEE**, chữ nội dung 8.2 pt ở kích thước xuất gốc. Không thu cả hình vào một cột vì chữ sẽ nhỏ. Waveform được vẽ bằng đường vector; không dán ảnh sóng raster vào sơ đồ.
'''),
        md(r'''## 1. Nội dung kiến trúc và đối chiếu mã nguồn

Nguồn cấu hình: `notebooks/kaggle_ablation_u4_all_noise_512_100ep.ipynb`, lớp `UNet1D` và `SelfAttention1D` được nhúng trong notebook đó. Bản tạo notebook vẽ đã đối chiếu mặc định 2 kênh vào, 80 kênh cơ sở, embedding 128 chiều, 1 kênh ra, attention 4 head và ba phép nối skip. Hash nguồn được lưu tại mục 2 để truy vết; Colab không cần đọc lại file nguồn.

| Khối | Thao tác | Đầu ra (kênh × chiều dài) |
|---|---|---|
| Đầu vào | Nối $x_s$ và ECG quan sát $y$ | $2\times512$ |
| Encoder 1 | HNF → FiLM | $80\times512$ |
| Encoder 2 | Down → HNF → FiLM | $160\times256$ |
| Encoder 3 | Down → HNF → FiLM | $320\times128$ |
| Encoder 4 | Down → HNF → FiLM | $640\times64$ |
| Attention | Self-attention 4 head | $640\times64$ |
| Decoder 3 | Up → nối e3 → HNF | $320\times128$ |
| Decoder 2 | Up → nối e2 → HNF | $160\times256$ |
| Decoder 1 | Up → nối e1 → HNF | $80\times512$ |
| Đầu ra | Conv1D kernel 1 | $1\times512$ |

- $s=\sqrt{\bar\alpha_t}$ đi qua embedding hình sin 128 chiều và điều kiện hóa **bốn encoder**, không phải decoder.
- Mỗi skip lấy đặc trưng encoder **sau FiLM**. Up thực hiện trước khi nối; sau nối có lần lượt 640, 320, 160 kênh rồi HNF giảm về 320, 160, 80.
- Down: Conv1D; Up: ConvTranspose1D; đều kernel 4, stride 2, padding 1.
- Đầu ra $\hat\epsilon$ là Gaussian dự đoán, **không phải ECG phục hồi trực tiếp**; không đặt sigmoid/tanh cuối mạng.
- Attention không có phép cộng residual riêng bên ngoài trong mã nguồn hiện hành.

Đây là sơ đồ phương pháp cho giao thức **512 điểm**, không phải hình kết quả thực nghiệm hay benchmark 3600 điểm.
'''),
        md('## 2. Cấu hình hình và vị trí khối\n\nCó thể đổi cỡ chữ, màu, độ nét và tọa độ. `GRAYSCALE=True` cho bản xám; `STRICT_TEXT_CHECK=True` kiểm tra khung/chữ. Mặc định `SIGNAL_MODE=illustration` không cần dữ liệu hoặc checkpoint.\n\nMuốn dùng tín hiệu thật: tải file `.npz` lên Files của Colab, đặt `SIGNAL_MODE=provided` và `SIGNALS_NPZ` thành đường dẫn file. Cần ba vector hữu hạn dài 512: `y` (ECG nhiễu quan sát), `x_s` (trạng thái diffusion) và `epsilon_pred` (Gaussian do mạng dự đoán), cùng một mẫu/mức diffusion. Notebook không tự tạo dữ liệu thay thế nếu thiếu key. Dùng `symbols` để tắt waveform.'),
        code(config),
        md('## 3. Hàm vẽ cơ bản\n\nCác khối bo góc biểu diễn một mức xử lý. Mũi tên đen cong tạo luồng chữ U; xanh thẳng là skip; vàng đứt là tham chiếu embedding cho FiLM. Không cần sửa cell này nếu chỉ thay nội dung/đường đi của sơ đồ.'),
        code(helpers),
        md('## 4. Vẽ kiến trúc chữ U đối xứng\n\nSửa chuỗi chữ hoặc tọa độ ở đây để chỉnh hình. Một khối bottleneck giữa đáy chứa Encoder 4 và attention. Không tách riêng Up/Concat thành nhiều ô nhỏ; thứ tự phép tính được ghi bên trong mỗi khối decoder. Điều kiện dùng tham chiếu ngắn $e(s)$ để tránh đường dây cắt qua skip. Không bỏ timestep hay chuyển FiLM sang decoder.'),
        code(drawing),
        md('## 5. Hiển thị hình ngay trong Colab\n\nChạy lại cell này sau khi sửa cấu hình/hàm vẽ. Hình tạo trong RAM và hiện ngay bên dưới, **không ghi file, không tự tải xuống**.'),
        code('''fig = make_architecture_figure()
try:
    with BytesIO() as buffer:
        fig.savefig(buffer, format='png', dpi=PREVIEW_DPI)
        display(Image(data=buffer.getvalue()))
finally:
    plt.close(fig)  # Không hiện thêm bản trùng cuối cell.
'''),
        md('## 6. Xuất và tải hình — tùy chọn\n\nMặc định Run all chỉ xem hình. Khi chốt bố cục, đặt `EXPORT_FILES=True` rồi chạy riêng cell dưới để xuất `2.png` (600 dpi), `2.pdf`, `2.svg`, bản mô tả và ZIP trong Files của Colab. Chỉ bật thêm `DOWNLOAD_ZIP=True` khi muốn tải về máy. Bước xuất vẽ lại từ cấu hình/hàm hiện hành nên không dùng nhầm preview cũ.'),
        code('''EXPORT_FILES = False
DOWNLOAD_ZIP = False

def export_architecture():
    figure = make_architecture_figure()
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in ('pdf', 'svg', 'png'):
            path = OUTPUT_DIR / f'{OUTPUT_BASENAME}.{extension}'
            options = {}
            if extension == 'pdf':
                options['metadata'] = {'Title': 'U4 architecture', 'CreationDate': None, 'ModDate': None}
            elif extension == 'svg':
                options['metadata'] = {'Date': None}
            else:
                options['dpi'] = PNG_DPI
            figure.savefig(path, **options)
            paths.append(path)
        specification = {
            'kind': 'method_diagram_not_experimental_result', 'figure': 2,
            'model': MODEL_SPEC, 'source_notebook': SOURCE_NOTEBOOK,
            'source_architecture_sha256': SOURCE_ARCHITECTURE_SHA256,
            'figure_inches': list(figure.get_size_inches()),
            'layout': 'symmetric_u_central_e4_attention_grouped_decoder',
            'signal_mode': SIGNAL_MODE,
            'signals_are_illustrative': SIGNAL_MODE == 'illustration',
            'illustration_seed': ILLUSTRATION_SEED if SIGNAL_MODE == 'illustration' else None,
            'signal_source': str(SIGNALS_NPZ) if SIGNAL_MODE == 'provided' else None,
            'font_scale': FONT_SCALE, 'grayscale': GRAYSCALE, 'png_dpi': PNG_DPI,
            'encoder_x': ENCODER_X, 'decoder_x': DECODER_X, 'level_y': LEVEL_Y,
            'bottleneck_y': BOTTLENECK_Y,
            'skip_sources': 'e1/e2/e3 after FiLM',
            'conditioning': 'sinusoidal embedding -> FiLM at all four encoders',
        }
        spec_path = OUTPUT_DIR / 'figure_specification.json'
        spec_path.write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding='utf-8')
        paths.append(spec_path)
        zip_path = OUTPUT_DIR / 'fig02_architecture.zip'
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, arcname=path.name)
        for path in paths:
            print(path.name, f'({path.stat().st_size:,} bytes)')
        print('ZIP:', zip_path)
        return zip_path
    finally:
        plt.close(figure)

if EXPORT_FILES:
    zip_path = export_architecture()
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
        md(r'''## 7. Chèn vào paper sau khi chốt hình

Trên Overleaf, tải `2.png` lên thư mục gốc và chèn bằng tên file `{2.png}` (hoặc dùng `{2.pdf}` cho vector). Bản cục bộ có thể lưu trong `output/imgs/` nhờ `graphicspath`. Notebook này **không tự sửa paper** và không thay các ảnh Hình 1 hiện có.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{2.png}
  \caption{Kiến trúc HNF--FiLM 1D U-Net (U4) cho đoạn ECG 512 điểm.
  Khối giữa đáy gom Encoder 4 và attention; mỗi decoder gom Up, nối skip và HNF.
  Các skip lấy đặc trưng sau FiLM; các nhãn e(s) chỉ cùng embedding điều kiện
  cho cả bốn encoder. Mạng dự đoán nhiễu Gaussian, không phải ECG phục hồi trực tiếp.
  Các waveform tổng hợp chỉ minh họa loại tín hiệu, không phải kết quả thực nghiệm.}
  \label{fig:architecture}
\end{figure*}
```

Trước khi chèn: kiểm tra cỡ chữ ở kích thước in thật, ba skip, bốn nhánh FiLM và các kích thước sau nối kênh. Kiểm tra hình trên Colab sau mỗi lần sửa vị trí khối; notebook không tự điều chỉnh các đường nối thủ công nếu thay đổi bố cục lớn.
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
    print(f'{DESTINATION}\nValidated {len(cells)} cells; model source checked.')


if __name__ == '__main__':
    main()
