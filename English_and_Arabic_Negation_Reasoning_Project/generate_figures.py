import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DARK = '#2f2f2f'
MID = '#7a7a7a'
LIGHT = '#c7c7c7'
GRID = '#e1e1e1'
WHITE = '#ffffff'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.edgecolor': '#9a9a9a',
    'axes.labelcolor': DARK,
    'text.color': DARK,
    'xtick.color': DARK,
    'ytick.color': DARK,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
    'savefig.facecolor': WHITE,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='result')
    p.add_argument('--output', default='figures')
    return p.parse_args()


def read_csv(base: Path, rel: str) -> pd.DataFrame:
    path = base / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def name_model(s):
    return 'Granite' if 'granite' in str(s).lower() else 'Qwen'


def canvas(figsize=(10.5, 5.8), bottom=0.14, left=0.10, right=0.98, top=0.97):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', length=0)
    return fig, ax


def save(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.18)
    fig.savefig(out_dir / f'{name}.svg', bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)


def annotated_bars(ax, bars, fmt='{:.1f}', dy=0.8, fontsize=8.5):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h),
                ha='center', va='bottom', fontsize=fontsize, color=DARK)


def barstyle(ax, ymax=100, ylabel='Accuracy (%)'):
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_yticks(np.arange(0, ymax + 1, 10))


def write_captions(out_dir: Path):
    captions = [
        ('Figure_01_English_Overall', 'Overall English accuracy for the reproduced runs versus the published paper results on the 1,100-statement English subset.'),
        ('Figure_02_English_Distractor', 'English distractor-negation accuracy for the reproduced runs versus the published paper results.'),
        ('Figure_03_Arabic_Test_Overall', 'Overall Arabic held-out test accuracy across the seven evaluated prompts for Granite and Qwen.'),
        ('Figure_04_Arabic_Polarity', 'Arabic held-out test accuracy split into affirmative and negated statements for the baseline prompt P1 and the proposed Arabic prompt P6.'),
        ('Figure_05_Arabic_Matched_Pairs', 'Matched-pair accuracy on the Arabic held-out test across the seven prompts.'),
        ('Figure_06_Negation_Categories', 'Granite accuracy by negated construction category on the Arabic held-out test comparing P1 and P6.'),
        ('Figure_07_Paired_Confidence_Intervals', 'Paired bootstrap confidence intervals for the P6 gains relative to P1 and P7 on the Arabic held-out test.'),
        ('Figure_08_Prediction_Bias', 'Predicted True rates under selected prompts compared with the balanced 50 percent reference prevalence.'),
        ('Figure_09_Robustness', 'Overall Arabic held-out test accuracy for the final P6 prompt, its paraphrases, and the reversed-order variant.'),
        ('Figure_10_Example_Origin', 'Arabic held-out test accuracy for translated versus Arabic-authored examples under P1 and P6.'),
        ('Figure_11_Development_Selection', 'Development-set accuracy used to select the final Arabic-specific prompt variant before the held-out test.'),
    ]
    lines = []
    for name, caption in captions:
        lines.append(f'{name}\n{caption}\n')
    (out_dir / 'Figure_Captions.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    args = parse_args()
    source = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    english = read_csv(source, 'English/english_comparison.csv')
    test = read_csv(source, 'Arabic_Test/test_summary.csv')
    rob = read_csv(source, 'Arabic_Robustness/robustness_summary.csv')
    ci = read_csv(source, 'Arabic_Test/confidence_intervals.csv')
    breakdown = read_csv(source, 'Arabic_Test/test_breakdown.csv')
    dev = read_csv(source, 'Arabic_Development/development_summary.csv')

    for frame in [english, test, rob, ci, breakdown, dev]:
        frame['model_label'] = frame['model'].map(name_model)

    ordering = ['BASELINE', 'WP1', 'WP2', 'PP1', 'PP2']
    english_map = [
        ('overall', 'Figure_01_English_Overall', 74),
        ('distractor_negation', 'Figure_02_English_Distractor', 43),
    ]
    for metric, fname, ymax in english_map:
        fig, ax = canvas(figsize=(11.7, 5.8), bottom=0.18)
        x = np.concatenate([np.arange(5), np.arange(5) + 6])
        rows = []
        for m in ['Granite', 'Qwen']:
            sub = english[english.model_label == m].set_index('prompt').loc[ordering]
            rows.append(sub)
        rows = pd.concat(rows)
        vals_pub = rows[f'{metric}_published_pct'].to_numpy()
        vals_rep = rows[f'{metric}_reproduced_pct'].to_numpy()
        b1 = ax.bar(x - 0.20, vals_pub, 0.37, color=LIGHT, label='Published', edgecolor=WHITE)
        b2 = ax.bar(x + 0.20, vals_rep, 0.37, color=DARK, label='Reproduced', edgecolor=WHITE)
        annotated_bars(ax, b2, dy=0.7)
        ax.axvline(5, color=MID, lw=0.9, ls=(0, (3, 4)))
        ax.set_xticks(x, ordering * 2)
        ymax = max(ymax, np.ceil(max(vals_pub.max(), vals_rep.max()) / 10) * 10 + 8)
        ax.set_ylim(0, ymax)
        ax.set_ylabel('Accuracy (%)')
        ax.legend(frameon=False, ncol=2, loc='upper right', fontsize=10)
        ax.text(2, -0.18, 'GRANITE', ha='center', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=10)
        ax.text(8, -0.18, 'QWEN', ha='center', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=10)
        save(fig, out_dir, fname)

    order = [f'P{i}' for i in range(1, 8)]
    pivot = test.pivot(index='prompt_id', columns='model_label', values='overall_accuracy_pct').loc[order]
    fig, ax = canvas(figsize=(11.0, 5.7), bottom=0.16)
    x = np.arange(7)
    w = 0.35
    b1 = ax.bar(x - w / 2, pivot['Granite'], w, color=DARK, label='Granite')
    b2 = ax.bar(x + w / 2, pivot['Qwen'], w, color=LIGHT, label='Qwen', edgecolor=MID, linewidth=0.5)
    annotated_bars(ax, b1, dy=0.4)
    annotated_bars(ax, b2, dy=0.4)
    ax.set_xticks(x, ['P1\nBase', 'P2', 'P3', 'P4', 'P5', 'P6\nProp.', 'P7\nCtrl.'])
    barstyle(ax, 75)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, out_dir, 'Figure_03_Arabic_Test_Overall')

    fig, ax = canvas(figsize=(10.8, 5.6), bottom=0.16)
    labels = ['Granite\nAffirmative', 'Granite\nNegated', 'Qwen\nAffirmative', 'Qwen\nNegated']
    y1, y6 = [], []
    for m in ['Granite', 'Qwen']:
        for attr in ['affirmative_accuracy_pct', 'negated_accuracy_pct']:
            y1.append(float(test[(test.model_label == m) & (test.prompt_id == 'P1')][attr].iloc[0]))
            y6.append(float(test[(test.model_label == m) & (test.prompt_id == 'P6')][attr].iloc[0]))
    x = np.array([0, 1, 3, 4])
    w = 0.34
    b1 = ax.bar(x - w / 2, y1, w, color=LIGHT, label='P1', edgecolor=MID, linewidth=0.5)
    b2 = ax.bar(x + w / 2, y6, w, color=DARK, label='P6')
    annotated_bars(ax, b1, dy=0.6)
    annotated_bars(ax, b2, dy=0.6)
    ax.set_xticks(x, labels)
    barstyle(ax, 82)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, out_dir, 'Figure_04_Arabic_Polarity')

    pivot = test.pivot(index='prompt_id', columns='model_label', values='matched_pair_accuracy_pct').loc[order]
    fig, ax = canvas(figsize=(10.8, 5.7), bottom=0.15)
    x = np.arange(7)
    w = 0.35
    b1 = ax.bar(x - w / 2, pivot['Granite'], w, color=DARK, label='Granite')
    b2 = ax.bar(x + w / 2, pivot['Qwen'], w, color=LIGHT, label='Qwen', edgecolor=MID, linewidth=0.5)
    annotated_bars(ax, b1, dy=0.4)
    annotated_bars(ax, b2, dy=0.4)
    ax.set_xticks(x, order)
    barstyle(ax, 48)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, out_dir, 'Figure_05_Arabic_Matched_Pairs')

    lookup = [
        ('scope_quantifier', 'Scope / quantifier'),
        ('لا', 'La'),
        ('لا / scope & quantifiers', 'La + scope'),
        ('لم', 'Lam'),
        ('لن', 'Lan'),
        ('ليس', 'Laysa'),
    ]
    fig, ax = canvas(figsize=(11.0, 5.9), bottom=0.21)
    sub = breakdown[(breakdown.model_label == 'Granite') & (breakdown.prompt_id.isin(['P1', 'P6']))]
    vals1, vals6, ns = [], [], []
    for term, _label in lookup:
        a = sub[(sub.metric == 'negated_construction:' + term) & (sub.prompt_id == 'P1')]
        b = sub[(sub.metric == 'negated_construction:' + term) & (sub.prompt_id == 'P6')]
        if a.empty or b.empty:
            raise RuntimeError('Missing category ' + term)
        vals1.append(float(a.accuracy_pct.iloc[0]))
        vals6.append(float(b.accuracy_pct.iloc[0]))
        ns.append(int(a.n.iloc[0]))
    x = np.arange(len(lookup))
    w = 0.35
    b1 = ax.bar(x - w / 2, vals1, w, color=LIGHT, label='P1', edgecolor=MID, linewidth=0.5)
    b2 = ax.bar(x + w / 2, vals6, w, color=DARK, label='P6')
    annotated_bars(ax, b1, fmt='{:.0f}', dy=0.5)
    annotated_bars(ax, b2, fmt='{:.0f}', dy=0.5)
    ax.set_xticks(x, [f'{lab}\nn={n}' for (_, lab), n in zip(lookup, ns)])
    barstyle(ax, 100)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, out_dir, 'Figure_06_Negation_Categories')

    fig, ax = canvas(figsize=(11.7, 6.7), bottom=0.09, left=0.29, right=0.98)
    records = []
    metric_names = {'overall': 'Overall', 'negated': 'Negated', 'matched_pair': 'Matched pairs'}
    for m in ['Granite', 'Qwen']:
        for ref in ['P1', 'P7']:
            for metric in ['overall', 'negated', 'matched_pair']:
                r = ci[(ci.model_label == m) & (ci.candidate_prompt == 'P6') & (ci.reference_prompt == ref) & (ci.metric == metric)]
                if len(r) != 1:
                    raise RuntimeError((m, ref, metric, len(r)))
                row = r.iloc[0]
                records.append((m, ref, metric, float(row.gain_pp), float(row.ci95_low_pp), float(row.ci95_high_pp)))
    y = np.arange(len(records))[::-1]
    for idx, (m, ref, metric, g, lo, hi) in enumerate(records):
        ax.errorbar(g, y[idx], xerr=np.array([[g - lo], [hi - g]]), fmt='o',
                    color=DARK if m == 'Granite' else MID, ecolor=MID,
                    elinewidth=1.6, capsize=3, markersize=5)
        ax.text(hi + 0.6, y[idx], f'{g:+.2f}', ha='left', va='center', fontsize=8.5)
    ax.axvline(0, lw=1.2, linestyle='--', color=MID)
    ax.set_yticks(y, [f'{m}  P6 vs {ref}  |  {metric_names[metric]}' for m, ref, metric, *_ in records])
    ax.set_xlim(-19, 36)
    ax.set_xlabel('Accuracy difference (percentage points)')
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='y', labelsize=9)
    save(fig, out_dir, 'Figure_07_Paired_Confidence_Intervals')

    fig, ax = canvas(figsize=(10.8, 5.7), bottom=0.14)
    selected = ['P1', 'P6', 'P7']
    xs = np.arange(6)
    yval = []
    labels = []
    for m in ['Granite', 'Qwen']:
        for p in selected:
            r = test[(test.model_label == m) & (test.prompt_id == p)].iloc[0]
            yval.append(float(r.predicted_true_pct))
            labels.append(f'{m}\n{p}')
    colors = [DARK, MID, LIGHT, DARK, MID, LIGHT]
    bars = ax.bar(xs, yval, color=colors, edgecolor=MID, linewidth=0.45, width=0.63)
    annotated_bars(ax, bars, dy=1.0)
    ax.axhline(50, color=DARK, lw=1.2, ls='--', label='Reference 50%')
    ax.set_xticks(xs, labels)
    barstyle(ax, 110, 'Predicted True (%)')
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    save(fig, out_dir, 'Figure_08_Prediction_Bias')

    fig, ax = canvas(figsize=(10.9, 5.8), bottom=0.15)
    selected = ['P6', 'P6_R1', 'P6_R2', 'P6_R3', 'P6_REVERSED']
    rows = []
    for m in ['Granite', 'Qwen']:
        vals = []
        for p in selected:
            src = test if p == 'P6' else rob
            vals.append(float(src[(src.model_label == m) & (src.prompt_id == p)].overall_accuracy_pct.iloc[0]))
        rows.append(vals)
    x = np.arange(len(selected))
    w = 0.35
    b1 = ax.bar(x - w / 2, rows[0], w, color=DARK, label='Granite')
    b2 = ax.bar(x + w / 2, rows[1], w, color=LIGHT, label='Qwen', edgecolor=MID, linewidth=0.5)
    annotated_bars(ax, b1, dy=0.4)
    annotated_bars(ax, b2, dy=0.4)
    ax.set_xticks(x, ['P6\nOriginal', 'R1', 'R2', 'R3', 'P6\nReversed'])
    barstyle(ax, 75)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, out_dir, 'Figure_09_Robustness')

    fig, ax = canvas(figsize=(11.0, 5.7), bottom=0.15)
    rows = []
    for m in ['Granite', 'Qwen']:
        for attr in ['translated_accuracy_pct', 'arabic_authored_accuracy_pct']:
            a = float(test[(test.model_label == m) & (test.prompt_id == 'P1')][attr].iloc[0])
            b = float(test[(test.model_label == m) & (test.prompt_id == 'P6')][attr].iloc[0])
            rows.append((a, b))
    x = np.array([0, 1, 3, 4])
    w = 0.34
    b1 = ax.bar(x - w / 2, [v[0] for v in rows], w, color=LIGHT, label='P1', edgecolor=MID, linewidth=0.5)
    b2 = ax.bar(x + w / 2, [v[1] for v in rows], w, color=DARK, label='P6')
    annotated_bars(ax, b1, dy=0.5)
    annotated_bars(ax, b2, dy=0.5)
    ax.set_xticks(x, ['Granite\nTranslated', 'Granite\nArabic-authored', 'Qwen\nTranslated', 'Qwen\nArabic-authored'])
    barstyle(ax, 75)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, out_dir, 'Figure_10_Example_Origin')

    piv = dev[dev.prompt_id.isin(['P6', 'P6_V2', 'P6_V3', 'P6_V4'])].pivot(index='prompt_id', columns='model_label', values='overall_accuracy_pct').loc[['P6', 'P6_V2', 'P6_V3', 'P6_V4']]
    fig, ax = canvas(figsize=(10.8, 5.7), bottom=0.15)
    x = np.arange(4)
    w = 0.35
    b1 = ax.bar(x - w / 2, piv['Granite'], w, color=DARK, label='Granite')
    b2 = ax.bar(x + w / 2, piv['Qwen'], w, color=LIGHT, label='Qwen', edgecolor=MID, linewidth=0.5)
    annotated_bars(ax, b1, dy=0.5)
    annotated_bars(ax, b2, dy=0.5)
    ax.set_xticks(x, ['Original P6', 'P6 V2', 'P6 V3', 'P6 V4'])
    barstyle(ax, 75)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, out_dir, 'Figure_11_Development_Selection')

    write_captions(out_dir)
    print(f'Created {len(list(out_dir.glob("*.png")))} PNG files and {len(list(out_dir.glob("*.svg")))} SVG files in {out_dir}')


if __name__ == '__main__':
    main()
