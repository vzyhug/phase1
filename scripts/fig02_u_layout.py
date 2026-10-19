"""Compact, geometrically symmetric U4 schematic for the Colab notebook.

Encoder 4 and attention share the central bottom stage. Decoder stages group
Up -> Concat -> HNF; the skip port denotes the internal Concat, not the Up.
"""


def stage_box(ax, x, y, width, height, text, *, fill=WHITE, size=8.2):
    patch = FancyBboxPatch((x-width/2, y-height/2), width, height,
                           boxstyle='round,pad=0,rounding_size=0.12',
                           linewidth=0.8, edgecolor=INK, facecolor=fill, zorder=3)
    ax.add_patch(patch)
    artist = label(ax, x, y, text, size=size)
    if not hasattr(ax.figure, '_diagram_box_labels'):
        ax.figure._diagram_box_labels = []
    ax.figure._diagram_box_labels.append((patch, artist))
    return {'left': (x-width/2, y), 'right': (x+width/2, y),
            'top': (x, y+height/2), 'bottom': (x, y-height/2), 'center': (x, y)}


def u_arrow(ax, start, end, *, width=1.25):
    mid_y = (start[1] + end[1]) / 2
    path = MplPath([start, (start[0], mid_y), (end[0], mid_y), end],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=8,
                                linewidth=width, color=INK, zorder=2,
                                shrinkA=0, shrinkB=0))


def condition_port(ax, target):
    """Named references to one shared e(s), avoiding a wire across every skip."""
    x, y = target
    arrow(ax, [(x-1.3, y), target], color=GOLD, dashed=True, width=0.8)
    label(ax, x-0.85, y+0.43, r'$e(s)$', color=GOLD, size=8.0)


def io_signals():
    """Return explicitly illustrative traces or user-supplied model traces."""
    if SIGNAL_MODE == 'symbols':
        return None
    length = MODEL_SPEC['length']
    if SIGNAL_MODE == 'provided':
        if SIGNALS_NPZ is None:
            raise ValueError('SIGNAL_MODE=provided cần SIGNALS_NPZ; dùng illustration nếu chỉ vẽ minh họa.')
        traces = {}
        with np.load(Path(SIGNALS_NPZ), allow_pickle=False) as data:
            for key in ('y', 'x_s', 'epsilon_pred'):
                if key not in data:
                    raise ValueError(f'File thiếu {key}; cần y, x_s và epsilon_pred của cùng một mẫu.')
                values = np.asarray(data[key], dtype=float).squeeze()
                if values.ndim != 1 or len(values) != length or not np.isfinite(values).all():
                    raise ValueError(f'{key} phải là vector hữu hạn dài {length}; nhận {values.shape}.')
                traces[key] = values
        traces['notice'] = 'Tín hiệu từ file cung cấp; notebook chỉ vẽ, không chạy mô hình.'
        traces['output_label'] = r'$\hat\epsilon$ — Gaussian dự đoán'
        return traces
    if SIGNAL_MODE != 'illustration':
        raise ValueError('SIGNAL_MODE phải là illustration, provided hoặc symbols.')

    # An analytic cartoon ECG, not a physiological record or a training sample.
    # The output noise is only a visual stand-in; no model prediction is made.
    rng = np.random.RandomState(ILLUSTRATION_SEED)
    t = np.arange(length) / 360.0
    clean = np.zeros(length)
    for r in (0.35, 1.05):
        for amplitude, offset, width in (
            (0.12, -0.18, 0.025), (-0.16, -0.035, 0.009),
            (1.0, 0.0, 0.009), (-0.25, 0.04, 0.012), (0.28, 0.25, 0.06),
        ):
            clean += amplitude * np.exp(-0.5 * ((t-r-offset)/width)**2)
    observed = clean + 0.07*np.sin(2*np.pi*0.6*t) + 0.025*rng.standard_normal(length)
    epsilon = rng.standard_normal(length)
    s = 0.92
    state = s*clean + np.sqrt(1-s*s)*epsilon
    return {
        'y': observed, 'x_s': state,
        'epsilon_pred': rng.standard_normal(length),
        'notice': 'Tín hiệu tổng hợp minh họa — không phải kết quả thực nghiệm.',
        'output_label': r'$\hat\epsilon$ — Gaussian minh họa',
    }


def signal_axes(ax, bounds, values, color):
    """Draw an editable vector waveform inset, not an image pasted into the figure."""
    x0, y0, x1, y1 = bounds
    fig = ax.figure
    points = fig.transFigure.inverted().transform(ax.transData.transform([[x0, y0], [x1, y1]]))
    panel = fig.add_axes((points[0, 0], points[0, 1],
                         points[1, 0]-points[0, 0], points[1, 1]-points[0, 1]))
    panel.axhline(0, color='#D4DAE0', linewidth=0.4)
    panel.plot(np.arange(len(values)), values, color=color, linewidth=0.65)
    extent = max(float(np.max(np.abs(values))), 1e-6)
    panel.set(xlim=(0, len(values)-1), ylim=(-1.12*extent, 1.12*extent))
    panel.axis('off')  # Schematic signals: no clinical amplitude or measured metric implied.
    return panel


def draw_architecture(ax, traces=None):
    c, length = MODEL_SPEC['base_channels'], MODEL_SPEC['length']
    shape = lambda channels, size: f'{channels} × {size}'
    nodes, encoders, decoders = {}, [], []
    label(ax, 11, 19.65, 'HNF–FiLM 1D U-Net (U4)', size=11, weight='bold')

    if traces is None:
        inputs = stage_box(ax, ENCODER_X[0], 16.4, 4.8, 1.6,
                           'Đầu vào: ' + r'$[x_s,y]$' + '\nNối kênh: ' + shape(2, length))
        outputs = stage_box(ax, DECODER_X[0], 16.4, 4.8, 1.6,
                            'Conv1D, kernel 1\n' + r'$\hat\epsilon$' + ': ' + shape(1, length))
    else:
        label(ax, 11, 19.10, traces['notice'], size=7.8, color=MUTED)
        inputs = stage_box(ax, ENCODER_X[0], 16.6, 5.6, 4.2, '', fill=WHITE)
        outputs = stage_box(ax, DECODER_X[0], 16.6, 5.6, 4.2, '', fill=WHITE)
        label(ax, ENCODER_X[0], 18.3, 'Đầu vào: ' + r'$[x_s,y]$', size=8.2)
        label(ax, 1.6, 17.13, r'$y$', size=8.2)
        label(ax, 1.6, 15.78, r'$x_s$', size=8.2)
        signal_axes(ax, (1.95, 16.65, 6.4, 17.6), traces['y'], BLUE)
        signal_axes(ax, (1.95, 15.3, 6.4, 16.25), traces['x_s'], MUTED)
        label(ax, ENCODER_X[0], 14.85, 'Nối kênh: ' + shape(2, length), size=8.0)
        label(ax, DECODER_X[0], 18.3, traces['output_label'], size=8.0)
        signal_axes(ax, (15.7, 15.45, 20.3, 17.7), traces['epsilon_pred'], NOISE_COLOR)
        label(ax, DECODER_X[0], 14.85, 'Conv1D k=1; ' + shape(1, length), size=8.0)
    stage_box(ax, 11, 16.4, 6.0, 1.8,
              'Điều kiện chung\n'
              + r'$s=\sqrt{\bar\alpha_t}\ \rightarrow\ e(s)\in\mathbb{R}^{128}$'
              + '\nFiLM tại cả bốn encoder', fill=FILM_FILL)

    # Three mirrored resolution levels. E4 is represented at the bottom.
    for index, y in enumerate(LEVEL_Y):
        stage, channels, size = index+1, c*2**index, length//2**index
        encoder = stage_box(ax, ENCODER_X[index], y, STAGE_WIDTH, STAGE_HEIGHT,
                            f'Encoder {stage}\nHNF → FiLM(' + r'$e(s)$' + ')\n'
                            + shape(channels, size), fill=HNF_FILL)
        decoder = stage_box(ax, DECODER_X[index], y, STAGE_WIDTH, STAGE_HEIGHT,
                            f'Decoder {stage}\nUp ×2 → Concat → HNF\n'
                            + shape(channels, size), fill=DECODER_FILL)
        encoders.append(encoder)
        decoders.append(decoder)
        nodes[f'e{stage}'], nodes[f'd{stage}'] = encoder, decoder
        condition_port(ax, encoder['left'])
        # Arrow enters the decoder stage's internal Concat, not its Up.
        arrow(ax, [encoder['right'], decoder['left']], color=BLUE, width=1.0)
        label(ax, 11, y+0.40, f'skip e{stage}', color=BLUE, size=8.0)
        if index:
            u_arrow(ax, encoders[index-1]['bottom'], encoder['top'])
            u_arrow(ax, decoder['top'], decoders[index-1]['bottom'])
            mid_y = (encoders[index-1]['bottom'][1] + encoder['top'][1]) / 2
            label(ax, ENCODER_X[index]-1.4, mid_y, '↓2', size=8.0)

    # A central bottom block makes the U explicit and preserves all of E4.
    bottom = stage_box(ax, 11, BOTTLENECK_Y, 8.8, 2.0,
                       'Bottleneck / Encoder 4\n'
                       + 'HNF → FiLM(' + r'$e(s)$' + ') → Self-attention (4 head)\n'
                       + shape(c*8, length//8), fill=ATTN_FILL)
    nodes['e4_attention'] = bottom
    condition_port(ax, bottom['left'])
    # The two connectors attach symmetrically to the top corners of the base.
    left_base, right_base = (8.5, BOTTLENECK_Y+1.0), (13.5, BOTTLENECK_Y+1.0)
    u_arrow(ax, encoders[-1]['bottom'], left_base)
    u_arrow(ax, right_base, decoders[-1]['bottom'])
    label(ax, 6.4, (encoders[-1]['bottom'][1]+left_base[1])/2, '↓2', size=8.0)
    arrow(ax, [inputs['bottom'], encoders[0]['top']], width=1.25)
    arrow(ax, [decoders[0]['top'], outputs['bottom']], width=1.25)

    label(ax, 11, 0.78,
          '↓2: Conv1D; Up ×2: ConvTranspose1D (kernel 4, stride 2, padding 1).', size=7.8)
    label(ax, 11, 0.32,
          r'$e(s)$' + ': cùng embedding cho bốn FiLM; skip chỉ vào Concat. Kích thước: kênh × chiều dài.',
          size=7.8)
    return nodes


def make_architecture_figure():
    if FIGURE_WIDTH_IN <= 0 or FIGURE_HEIGHT_IN <= 0 or FONT_SCALE <= 0:
        raise ValueError('Kích thước hình và FONT_SCALE phải dương.')
    traces = io_signals()
    fig, ax = canvas(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN, 22, 20.3)
    try:
        fig._architecture_nodes = draw_architecture(ax, traces)
        fig._signal_mode = SIGNAL_MODE
        check_diagram(fig)
        return fig
    except Exception:
        plt.close(fig)
        raise
