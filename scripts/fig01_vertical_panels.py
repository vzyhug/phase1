"""Portrait Figure 1 panels embedded into the self-contained Colab notebook.

Drawing helpers and palette are supplied by the notebook builder. Coordinates
are local to each panel: x in [0, 10]; y extents are configured in the notebook.
This is method-diagram code, not a simulation or experimental result.
"""


def draw_panel_a(ax):
    label(ax, 5, 19.55, '(a) Chuẩn bị dữ liệu', size=9.2, weight='bold')
    qt = box(ax, 2.7, 18.25, 4.2, 1.3,
             'QTDB\nECG một kênh', fill=WHITE)
    nst = box(ax, 7.3, 18.25, 4.2, 1.3,
              'NSTDB\nBW / EM / MA', fill=WHITE)
    prep = box(ax, 2.7, 15.75, 4.2, 2.4,
               'Tách đoạn theo nhịp\nResample 360 Hz\nĐệm 512 điểm')
    mix = box(ax, 7.3, 15.75, 4.2, 2.4,
              'Chọn 1/7 tổ hợp\nBW / EM / MA\nCộng patch nhiễu', fill=HNF_FILL)
    pairs = box(ax, 5, 12.35, 8.4, 2.4,
                'Tạo cặp tham chiếu–nhiễu\n'
                + r'$n_{\rm mix}=m_1n_{BW}+m_2n_{EM}+m_3n_{MA}$' + '\n'
                + r'$y=x_0+\kappa n_{\rm mix}$' + '\n'
                + 'Scale hỗn hợp một lần', fill=HNF_FILL)
    norm = box(ax, 5, 8.9, 8.4, 1.8,
               'Chuẩn hóa chung ' + r'$(x_0,y)$' + '\nGiữ cùng hệ số scale')
    split = box(ax, 5, 5.9, 8.4, 2.0,
                'Cố định split, mask và mức nhiễu\nLưu mã mẫu để tái lập')
    output = box(ax, 5, 2.95, 8.4, 1.8,
                 'Cặp đầu ra ' + r'$(x_0,y)$' + '\nMỗi tín hiệu: '
                 + r'$1\times512$', fill=WHITE)
    arrow(ax, [qt['bottom'], prep['top']])
    arrow(ax, [nst['bottom'], mix['top']])
    arrow(ax, [prep['bottom'], (2.7, 14.0), (3.5, 14.0), (3.5, 13.55)])
    arrow(ax, [mix['bottom'], (7.3, 14.0), (6.5, 14.0), (6.5, 13.55)])
    label(ax, 2.2, 13.95, r'$x_0$', size=8.2)
    label(ax, 7.9, 13.95, r'$n_{\rm mix}$', size=8.2)
    for upper, lower in zip((pairs, norm, split), (norm, split, output)):
        arrow(ax, [upper['bottom'], lower['top']])
    label(ax, 5, 0.95,
          '7 tổ hợp không rỗng từ BW, EM, MA.\n'
          'Không trộn artifact với Gaussian diffusion.', size=7.8, color=MUTED)


def draw_panel_b(ax):
    label(ax, 5, 21.55, '(b) Huấn luyện có điều kiện', size=9.2, weight='bold')
    source = box(ax, 5, 19.9, 7.6, 1.6,
                 'Cặp ' + r'$(x_0,y)$; $\epsilon\sim\mathcal{N}(0,I)$'
                 + '\nLấy mẫu mức diffusion ' + r'$s$', fill=WHITE)
    forward = box(ax, 5, 17.4, 7.6, 1.8,
                  'Diffusion thuận\n' + r'$x_s=sx_0+\sqrt{1-s^2}\epsilon$')
    net = box(ax, 5, 14.4, 7.6, 2.2,
              'HNF–FiLM 1D U-Net (U4)\n'
              + r'Ghép $[x_s,y]$; FiLM($s$) tại encoder'
              + '\nHNF tại encoder/decoder\nAttention tại bottleneck', fill=HNF_FILL)
    prediction = box(ax, 5, 11.85, 7.6, 1.2,
                     'Dự đoán Gaussian\n' + r'$\hat\epsilon=\epsilon_\theta(x_s,y,s)$',
                     fill=WHITE)
    reconstruction = box(ax, 5, 9.4, 7.6, 1.8,
                         'Ước lượng ECG tham chiếu\n'
                         + r'$\hat x_0=(x_s-\sqrt{1-s^2}\hat\epsilon)/s$')
    loss = box(ax, 5, 6.4, 7.6, 2.4,
               'Loss kết hợp\n' + r'$\mathcal{L}=\mathcal{L}_\epsilon+0.1\mathcal{L}_f$'
               + '\nL1: ' + r'$\hat\epsilon$' + ' so với ' + r'$\epsilon$'
               + '\nSTFT: ' + r'$\hat x_0,x_0$' + '; trọng số diffusion SNR',
               fill=FILM_FILL)
    optimizer = box(ax, 5, 3.9, 7.6, 1.2,
                    'Lan truyền ngược\nAdam; cập nhật ' + r'$\theta$', fill=HNF_FILL)
    checkpoint = box(ax, 5, 1.55, 7.6, 1.5,
                     'Chọn bằng validation loss\nLưu mạng và EMA tương ứng', fill=WHITE)
    for upper, lower in zip(
        (source, forward, net, prediction, reconstruction, loss, optimizer),
        (forward, net, prediction, reconstruction, loss, optimizer, checkpoint),
    ):
        arrow(ax, [upper['bottom'], lower['top']])
    label(ax, 5.5, 16.0, r'$x_s$', size=8.2)
    arrow(ax, [source['right'], (9.45, 19.9), (9.45, 14.4), net['right']],
          color=GOLD, dashed=True)
    label(ax, 9.75, 17.1, r'Điều kiện $y,s$', rotation=90, size=7.8, color=GOLD)
    arrow(ax, [prediction['right'], (9.2, 11.85), (9.2, 6.8), (8.8, 6.8)])
    label(ax, 9.65, 9.1, r'$\hat\epsilon$ cho $\mathcal{L}_\epsilon$',
          rotation=90, size=7.8)
    arrow(ax, [optimizer['left'], (0.6, 3.9), (0.6, 14.4), net['left']],
          color=BLUE, dashed=True)
    label(ax, 0.22, 9.1, r'Lan truyền ngược; cập nhật $\theta$',
          rotation=90, size=7.8, color=BLUE)


def draw_panel_c(ax):
    label(ax, 5, 16.55, '(c) Khôi phục ECG bằng DDIM', size=9.2, weight='bold')
    checkpoint = box(ax, 5, 15.0, 7.6, 1.5,
                     'Nạp checkpoint đã chọn\nGiữ cố định trọng số mạng', fill=WHITE)
    inputs = box(ax, 5, 12.5, 7.6, 1.8,
                 'ECG quan sát ' + r'$y$' + '\nKhởi tạo '
                 + r'$x_T\sim\mathcal{N}(0,I)$', fill=WHITE)
    prediction = box(ax, 5, 9.7, 7.6, 1.8,
                     'Dự đoán tại mốc ' + r'$\tau$' + '\n'
                     + r'$\hat\epsilon_\tau=\epsilon_\theta(x_\tau,y,s_\tau)$',
                     fill=HNF_FILL)
    update = box(ax, 5, 7.0, 7.6, 1.8,
                 'Cập nhật trạng thái bằng DDIM\n' + r'$x_\tau\rightarrow x_{\tau^\prime}$',
                 fill=HNF_FILL)
    repeat = box(ax, 5, 4.6, 7.6, 1.5,
                 'Còn mốc DDIM tiếp theo?\nLặp với cùng mạng và cùng ' + r'$y$')
    output = box(ax, 5, 1.95, 7.6, 1.6,
                 'ECG phục hồi\n' + r'$\hat x_0\in\mathbb{R}^{1\times512}$', fill=WHITE)
    for upper, lower in zip((checkpoint, inputs, prediction, update),
                            (inputs, prediction, update, repeat)):
        arrow(ax, [upper['bottom'], lower['top']])
    arrow(ax, [repeat['bottom'], output['top']])
    label(ax, 5.8, 3.35, 'Không', size=8.2)
    arrow(ax, [repeat['right'], (9.45, 4.6), (9.45, 9.7), prediction['right']],
          color=BLUE, dashed=True)
    label(ax, 9.75, 7.15, 'Có: sang mốc tiếp theo', rotation=90, size=7.8, color=BLUE)
    label(ax, 5, 0.45, 'Gaussian diffusion ≠ artifact BW/EM/MA.', size=7.8, color=MUTED)
