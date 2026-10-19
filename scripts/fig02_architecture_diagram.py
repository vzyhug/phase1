"""U4 diagram functions embedded into the standalone Colab notebook.

The builder supplies drawing helpers, palette, geometry, and the source-checked
model specification. No Torch model or ECG signal is executed by these functions.
"""


def draw_architecture(ax):
    c, length = MODEL_SPEC['base_channels'], MODEL_SPEC['length']
    emb_dim, heads = MODEL_SPEC['embedding_dim'], MODEL_SPEC['attention_heads']
    left, right = ENCODER_X, DECODER_X
    stage_width, stage_height = 5.2, 2.2
    shapes = [(c * 2 ** i, length // 2 ** i) for i in range(4)]
    shape = lambda channels, size: f'{channels} × {size}'
    nodes = {}

    label(ax, 9, 25.65, 'HNF–FiLM 1D U-Net (U4)',
          size=10.0, weight='bold')

    # 1. Two signal inputs are concatenated; s is not a signal channel.
    state = box(ax, left - 1.4, 23.8, 2.4, 1.2,
                r'$x_s$' + '\n' + shape(1, length), fill=WHITE)
    observed = box(ax, left + 1.4, 23.8, 2.4, 1.2,
                   r'$y$' + '\n' + shape(1, length), fill=WHITE)
    joined = box(ax, left, 21.5, 4.6, 1.3,
                 'Nối kênh ' + r'$[x_s,y]$' + '\n' + shape(2, length))
    arrow(ax, [state['bottom'], (left - 1.4, 22.6),
               (left - 0.7, 22.6), (left - 0.7, 22.15)])
    arrow(ax, [observed['bottom'], (left + 1.4, 22.6),
               (left + 0.7, 22.6), (left + 0.7, 22.15)])

    level = box(ax, 9.2, 24.2, 3.6, 1.0,
                r'$s=\sqrt{\bar\alpha_t}$', fill=FILM_FILL)
    embedding = box(ax, 9.2, 22.1, 3.6, 1.8,
                    'Embedding hình sin\n' + f'{emb_dim} chiều\nĐiều kiện cho 4 FiLM',
                    fill=FILM_FILL)
    arrow(ax, [level['bottom'], embedding['top']], color=GOLD, dashed=True)
    # Route the shared condition bus outside the feature/skip paths.
    line(ax, [embedding['left'], (7.1, 22.1), (7.1, 25.0),
              (0.65, 25.0), (0.65, ENCODER_Y[-1])], color=GOLD, dashed=True)
    label(ax, 0.2, 12.5, 'Embedding → FiLM tại encoder', rotation=90,
          size=8.0, color=GOLD)

    # 2. Four encoder levels. All skip sources are AFTER FiLM.
    encoders = []
    for index, ((channels, size), y) in enumerate(zip(shapes, ENCODER_Y)):
        title = f'Encoder {index + 1}' if index < 3 else 'Encoder 4 / bottleneck'
        encoder = box(ax, left, y, stage_width, stage_height,
                      title + '\nHNF → FiLM(' + r'$s$' + ')\n'
                      + shape(channels, size), fill=HNF_FILL)
        encoders.append(encoder)
        nodes[f'e{index + 1}'] = encoder
        arrow(ax, [(0.65, y), encoder['left']], color=GOLD, dashed=True)
        if index == 0:
            arrow(ax, [joined['bottom'], encoder['top']])
        else:
            upper = encoders[index - 1]
            down = box(ax, left, (ENCODER_Y[index - 1] + y) / 2,
                       3.8, 1.2, 'Down ×2\n' + shape(channels, size))
            arrow(ax, [upper['bottom'], down['top']])
            arrow(ax, [down['bottom'], encoder['top']])

    attention = box(ax, 9.3, ENCODER_Y[-1], 3.6, 1.9,
                    f'Self-attention\n{heads} head\n' + shape(*shapes[-1]),
                    fill=ATTN_FILL)
    arrow(ax, [encoders[-1]['right'], attention['left']])

    # 3. Decoder goes upward: Up -> Concat -> HNF at every level.
    previous = attention
    for index, stage in enumerate((3, 2, 1)):
        channels, size = shapes[stage - 1]
        up_y = UPSAMPLE_Y[index]
        concat_y = CONCAT_Y[index]
        decoder_y = DECODER_Y[index]
        up = box(ax, right, up_y, 4.6, 1.2,
                 'Up ×2\n' + shape(channels, size))
        if index == 0:
            arrow(ax, [previous['right'], (right, ENCODER_Y[-1]), up['bottom']])
        else:
            arrow(ax, [previous['top'], up['bottom']])
        concat = box(ax, right, concat_y, 0.6, 0.75, 'C',
                     fill=WHITE, size=7.6, weight='bold')
        arrow(ax, [up['top'], concat['bottom']])
        label(ax, right + 2.3, concat_y, 'Sau nối\n' + shape(2 * channels, size), size=8.0)

        skip = encoders[stage - 1]
        skip_y = skip['center'][1]
        arrow(ax, [skip['right'], (SKIP_BEND_X, skip_y),
                   (SKIP_BEND_X, concat_y), concat['left']],
              color=BLUE, dashed=True)
        label(ax, 10.8, concat_y + 0.42,
              f'skip e{stage}: ' + shape(channels, size), size=8.0, color=BLUE)

        decoder = box(ax, right, decoder_y, stage_width, 1.8,
                      f'Decoder {stage}\nHNF\n' + shape(channels, size), fill=HNF_FILL)
        arrow(ax, [concat['top'], decoder['bottom']])
        nodes[f'd{stage}'] = decoder
        previous = decoder

    projection = box(ax, right, 21.2, 4.6, 1.3,
                     'Conv1D, kernel 1\n' + f'{c} → 1 kênh')
    output = box(ax, right, 23.65, 4.6, 1.4,
                 r'$\hat\epsilon=\epsilon_\theta(x_s,y,s)$' + '\n' + shape(1, length),
                 fill=WHITE)
    arrow(ax, [previous['top'], projection['bottom']])
    arrow(ax, [projection['top'], output['bottom']])

    # Legend is part of the figure, not encoded only by color.
    label(ax, 9, 1.1,
          'C: nối kênh; Down/Up: kernel 4, stride 2, padding 1.', size=8.0)
    label(ax, 9, 0.6,
          'Nét liền: đặc trưng | Xanh đứt: skip | Vàng đứt: điều kiện FiLM.', size=8.0)
    label(ax, 9, 0.10,
          'Kích thước: kênh × chiều dài (bỏ batch). Đầu ra dự đoán Gaussian, không phải ECG.',
          size=7.8)
    return nodes


def check_diagram(fig):
    """Catch out-of-box text and labels outside the canvas at actual print size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    warnings = []
    for patch, artist in getattr(fig, '_diagram_box_labels', []):
        frame = patch.get_window_extent(renderer)
        text = artist.get_window_extent(renderer)
        if (text.x0 < frame.x0 + 1 or text.x1 > frame.x1 - 1
                or text.y0 < frame.y0 + 1 or text.y1 > frame.y1 - 1):
            warnings.append('Chữ vượt khối: ' + artist.get_text().replace('\n', ' / '))
    boxes = getattr(fig, '_diagram_box_labels', [])
    for index, (first, first_text) in enumerate(boxes):
        first_box = first.get_window_extent(renderer)
        for second, second_text in boxes[index + 1:]:
            second_box = second.get_window_extent(renderer)
            width = min(first_box.x1, second_box.x1) - max(first_box.x0, second_box.x0)
            height = min(first_box.y1, second_box.y1) - max(first_box.y0, second_box.y0)
            if width > 0.5 and height > 0.5:
                warnings.append('Hai khối chồng nhau: '
                                + first_text.get_text().replace('\n', ' / ') + ' & '
                                + second_text.get_text().replace('\n', ' / '))
    bounds = fig.bbox
    for axis in fig.axes:
        for artist in axis.texts:
            text = artist.get_window_extent(renderer)
            if (text.x0 < bounds.x0 or text.x1 > bounds.x1
                    or text.y0 < bounds.y0 or text.y1 > bounds.y1):
                warnings.append('Chữ vượt canvas: ' + artist.get_text().replace('\n', ' / '))
    for warning in warnings:
        print(warning)
    if warnings and STRICT_TEXT_CHECK:
        raise ValueError('Cần chỉnh khung/ngắt dòng hoặc tăng kích thước hình.')
    if not warnings:
        print('Kiểm tra chữ: nằm trong khung và canvas.')
    return warnings


def make_architecture_figure():
    if FIGURE_WIDTH_IN <= 0 or FIGURE_HEIGHT_IN <= 0 or FONT_SCALE <= 0:
        raise ValueError('Kích thước hình và FONT_SCALE phải dương.')
    fig, ax = canvas(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN, 18, 26.3)
    try:
        draw_architecture(ax)
        check_diagram(fig)
        return fig
    except Exception:
        plt.close(fig)
        raise
