"""Reference-inspired U-shaped diagram, preserving the actual U4 computation.

All functions are embedded by the notebook builder. Reference JPEG is a style
reference only; no reference image or trained predictions are required at run time.
"""


def rounded_box(ax, x, y, width, height, text, *, fill=WHITE, size=8.2, weight='normal'):
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                           boxstyle='round,pad=0,rounding_size=0.18',
                           linewidth=0.8, edgecolor=INK, facecolor=fill, zorder=3)
    ax.add_patch(patch)
    artist = label(ax, x, y, text, size=size, weight=weight)
    if not hasattr(ax.figure, '_diagram_box_labels'):
        ax.figure._diagram_box_labels = []
    ax.figure._diagram_box_labels.append((patch, artist))
    return {'left': (x-width/2, y), 'right': (x+width/2, y),
            'top': (x, y+height/2), 'bottom': (x, y-height/2), 'center': (x, y)}


def curve_arrow(ax, start, control1, control2, end, *, color=INK, width=1.1, head=8):
    path = MplPath([start, control1, control2, end],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=head,
                                linewidth=width, color=color, zorder=2,
                                shrinkA=0, shrinkB=0))


def draw_architecture(ax):
    c, length = MODEL_SPEC['base_channels'], MODEL_SPEC['length']
    shape = lambda channels, size: f'{channels} × {size}'
    encoders, nodes = [], {}
    if SHOW_OUTER_FRAME:
        ax.add_patch(Rectangle((0.08, 0.25), 20.84, 14.7, fill=False,
                               edgecolor=INK, linewidth=0.9, zorder=0))
    label(ax, 10.5, 19.3, 'HNF–FiLM 1D U-Net Noise Predictor', size=11, weight='bold')

    inputs = rounded_box(ax, ENCODER_X[0], 17.3, 4.1, 1.35,
                         r'$x_s,\;y$' + '\nNối kênh: ' + shape(2, length))
    output = rounded_box(ax, DECODER_X[-1], 17.3, 4.1, 1.35,
                         r'$\hat\epsilon=\epsilon_\theta(x_s,y,s)$' + '\n' + shape(1, length))
    condition = rounded_box(ax, 9.1, 16.9, 5.1, 1.9,
                            'Bridge / FiLM\n' + r'$s=\sqrt{\bar\alpha_t}$'
                            + f'\nEmbedding {MODEL_SPEC["embedding_dim"]} chiều',
                            fill=FILM_FILL)
    # A shared deterministic embedding value feeds four encoder FiLM blocks.
    # No decoder Bridge/FiLM is introduced from the reference illustration.
    xs = [condition['left'][0], 6.2, 6.2, 0.38, 0.38]
    ys = [condition['left'][1], 16.9, 18.4, 18.4, ENCODER_Y[-1]]
    ax.plot(xs, ys, color=INK, linewidth=0.85, linestyle=(0, (1.2, 1.8)), zorder=1)

    for index, (x, y) in enumerate(zip(ENCODER_X, ENCODER_Y)):
        channels, size = c * 2 ** index, length // 2 ** index
        encoder = rounded_box(ax, x, y, 3.6, 1.9,
                              f'Encoder {index+1}\nHNF → FiLM(' + r'$s$' + ')\n'
                              + shape(channels, size), fill=ENCODER_FILL)
        encoders.append(encoder)
        nodes[f'e{index+1}'] = encoder
        arrow(ax, [(0.38, y), encoder['left']], color=INK, dashed=True, width=0.8)
        if index == 0:
            arrow(ax, [inputs['bottom'], encoder['top']], width=1.7, head=10)
        else:
            previous = encoders[index - 1]
            curve_arrow(ax, previous['bottom'],
                        (previous['center'][0], y+0.5), (x-2.2, y), encoder['left'])
            label(ax, x-2.1, y+1.55, 'Down ×2', size=7.8)

    attention = rounded_box(ax, 10.9, 2.0, 4.4, 2.0,
                            'Bottleneck\nSelf-attention, '
                            + f'{MODEL_SPEC["attention_heads"]} head\n' + shape(c*8, length//8),
                            fill=ATTN_FILL, size=8.0)
    curve_arrow(ax, encoders[-1]['bottom'], (6.7, 2.0),
                (7.4, 2.0), attention['left'])

    previous = attention
    for index, stage in enumerate((3, 2, 1)):
        x, y = DECODER_X[index], DECODER_Y[index]
        channels, size = c * 2 ** (stage-1), length // 2 ** (stage-1)
        concat_x = x - 2.5
        concat = rounded_box(ax, concat_x, y, 0.65, 0.7, 'C', size=7.5, weight='bold')
        if index == 0:
            curve_arrow(ax, previous['right'], (concat_x+1.5, 2.0),
                        (concat_x, y-1.6), concat['bottom'])
            label(ax, concat_x+1.8, y-2.1, 'Up ×2\n' + shape(channels, size), size=7.8)
        else:
            curve_arrow(ax, previous['top'],
                        (previous['center'][0], y-1.5), (concat_x, y-1.5), concat['bottom'])
            label(ax, concat_x-1.2, y-1.5, 'Up ×2', size=7.8)
        decoder = rounded_box(ax, x, y, 3.4, 1.9,
                              f'Decoder {stage}\nHNF\n' + shape(channels, size),
                              fill=DECODER_FILL)
        arrow(ax, [concat['right'], decoder['left']], width=0.9)
        skip = encoders[stage-1]
        arrow(ax, [skip['right'], concat['left']], color=SKIP_COLOR, width=1.25)
        middle = (skip['right'][0] + concat['left'][0]) / 2
        label(ax, middle, y+0.44,
              f'skip e{stage}: ' + shape(channels, size), color=SKIP_COLOR, size=8.0)
        label(ax, concat_x-0.55, y-0.65, shape(channels*2, size), size=7.5, ha='right')
        nodes[f'd{stage}'] = decoder
        previous = decoder

    projection = rounded_box(ax, DECODER_X[-1], 15.55, 3.4, 0.85,
                             'Conv1D, kernel 1', size=8.0)
    arrow(ax, [previous['top'], projection['bottom']], width=1.1)
    arrow(ax, [projection['top'], output['bottom']], width=1.7, head=10)

    # Compact legend inside the reference-style frame; no fabricated curves.
    label(ax, 16.9, 4.7, 'CHÚ GIẢI', size=8.4, weight='bold')
    arrow(ax, [(14.5, 3.95), (15.5, 3.95)], color=SKIP_COLOR, width=1.25)
    label(ax, 15.8, 3.95, 'Skip sau FiLM', size=8.0, ha='left')
    arrow(ax, [(14.5, 3.2), (15.5, 3.2)], width=1.1)
    label(ax, 15.8, 3.2, 'Luồng đặc trưng', size=8.0, ha='left')
    arrow(ax, [(14.5, 2.45), (15.5, 2.45)], dashed=True, width=0.8)
    label(ax, 15.8, 2.45, 'Điều kiện FiLM', size=8.0, ha='left')
    label(ax, 16.9, 1.5, 'C: nối kênh\nDown/Up: k=4, s=2, p=1', size=7.8)
    label(ax, 10.5, 0.63, 'Kích thước: kênh × chiều dài; bỏ batch. Đầu ra là Gaussian dự đoán.',
          size=7.8)
    return nodes


def read_signal_panels():
    """Load supplied traces only; never simulate or infer a model prediction."""
    if SIGNALS_NPZ is None:
        return None
    path = Path(SIGNALS_NPZ)
    if not path.is_file():
        raise FileNotFoundError(f'Chưa có file tín hiệu: {path}. Để SIGNALS_NPZ=None nếu chỉ vẽ kiến trúc.')
    traces = {}
    with np.load(path, allow_pickle=False) as data:
        for key in ('x_s', 'y', 'epsilon', 'epsilon_pred'):
            if key not in data:
                raise ValueError(f'File thiếu {key}; cần x_s, y, epsilon, epsilon_pred của cùng một mẫu/mức s.')
            values = np.asarray(data[key], dtype=float).squeeze()
            if values.ndim != 1 or len(values) != MODEL_SPEC['length'] or not np.isfinite(values).all():
                raise ValueError(f'{key} phải là một vector hữu hạn dài 512; nhận {values.shape}.')
            traces[key] = values
    return traces


def draw_real_signals(fig, ax, traces):
    for bounds, keys, names, title in (
        ((0.9, 21.2, 7.0, 25.4), ('x_s', 'y'), (r'$x_s$', r'$y$'), 'Đầu vào — dữ liệu cung cấp'),
        ((13.1, 21.2, 20.2, 25.4), ('epsilon', 'epsilon_pred'),
         (r'Gaussian thật $\epsilon$', r'Dự đoán $\hat\epsilon$'), 'Đối chiếu Gaussian — dữ liệu cung cấp'),
    ):
        x0, y0, x1, y1 = bounds
        points = fig.transFigure.inverted().transform(ax.transData.transform([[x0, y0], [x1, y1]]))
        plot = fig.add_axes((points[0, 0], points[0, 1],
                             points[1, 0]-points[0, 0], points[1, 1]-points[0, 1]))
        axis = np.arange(MODEL_SPEC['length'])
        plot.plot(axis, traces[keys[0]], color=MUTED, linestyle='--', linewidth=0.65, label=names[0])
        plot.plot(axis, traces[keys[1]], color=TRACE_COLOR, linewidth=0.75, label=names[1])
        plot.set_title(title, fontsize=8.0 * FONT_SCALE)
        plot.set_xlabel('Điểm mẫu', fontsize=7.2 * FONT_SCALE)
        plot.set_ylabel('Biên độ chuẩn hóa', fontsize=7.2 * FONT_SCALE)
        plot.tick_params(labelsize=6.8 * FONT_SCALE)
        plot.grid(alpha=0.2, linewidth=0.5)
        plot.legend(fontsize=6.7 * FONT_SCALE, loc='upper right')


def make_architecture_figure():
    if FIGURE_WIDTH_IN <= 0 or FIGURE_HEIGHT_IN <= 0 or FONT_SCALE <= 0:
        raise ValueError('Kích thước hình và FONT_SCALE phải dương.')
    traces = read_signal_panels()
    ymax = 26.8 if traces is not None else 20.0
    height = FIGURE_HEIGHT_IN * ymax / 20.0
    fig, ax = canvas(FIGURE_WIDTH_IN, height, 21, ymax)
    try:
        draw_architecture(ax)
        if traces is not None:
            draw_real_signals(fig, ax, traces)
        check_diagram(fig)
        return fig
    except Exception:
        plt.close(fig)
        raise
