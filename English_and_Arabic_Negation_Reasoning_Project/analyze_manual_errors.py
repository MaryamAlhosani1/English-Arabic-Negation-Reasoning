import argparse
import csv
from collections import Counter
from pathlib import Path


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def model_name(full_name):
    if 'granite' in full_name.lower():
        return 'Granite'
    if 'qwen' in full_name.lower():
        return 'Qwen'
    raise ValueError(f'Unexpected model: {full_name}')


def main():
    parser = argparse.ArgumentParser(description='Reproduce the descriptive human error-review breakdown from the approved 50-case CSV.')
    parser.add_argument('--input', type=Path, default=Path('result/Arabic_Test/manual_error_review.csv'))
    parser.add_argument('--output', type=Path, default=Path('result/Arabic_Test/manual_error_analysis.csv'))
    parser.add_argument('--notes', type=Path, default=Path('manual_review/Error_Analysis.md'))
    args = parser.parse_args()

    rows = read_rows(args.input)
    if len(rows) != 50:
        raise ValueError(f'Expected 50 manually reviewed cases; found {len(rows)}')
    expected = {'model', 'item_id', 'statement_ar', 'label', 'baseline_prediction',
                'proposed_prediction', 'change', 'reviewer', 'error_category', 'explanation'}
    if not expected.issubset(rows[0]):
        raise ValueError(f'Missing columns: {expected - rows[0].keys()}')
    if 'review_id' in rows[0] or 'review_status' in rows[0]:
        raise ValueError('This approved export must not contain review_id or review_status')

    seen = set()
    for r in rows:
        key = (r['model'], r['item_id'])
        if key in seen:
            raise ValueError(f'Duplicate case: {key}')
        seen.add(key)
        if r['reviewer'] not in ('Munira', 'Maryam'):
            raise ValueError(f'Unrecognized reviewer: {r["reviewer"]}')
        if not r['error_category'].strip() or not r['explanation'].strip():
            raise ValueError(f'Empty interpretation: {key}')
        if r['change'] == 'corrected':
            if not (r['baseline_prediction'] != r['label'] and r['proposed_prediction'] == r['label']):
                raise ValueError(f'Inconsistent correction: {key}')
        elif r['change'] == 'introduced_error':
            if not (r['baseline_prediction'] == r['label'] and r['proposed_prediction'] != r['label']):
                raise ValueError(f'Inconsistent introduced error: {key}')
        else:
            raise ValueError(f'Unexpected change: {key}')

    reviewers = Counter(r['reviewer'] for r in rows)
    changes = Counter(r['change'] for r in rows)
    models = Counter(model_name(r['model']) for r in rows)
    if reviewers != {'Munira': 25, 'Maryam': 25}:
        raise ValueError(f'Unexpected reviewer balance: {reviewers}')
    if changes != {'corrected': 25, 'introduced_error': 25}:
        raise ValueError(f'Unexpected change balance: {changes}')
    if models != {'Granite': 26, 'Qwen': 24}:
        raise ValueError(f'Unexpected model balance: {models}')

    categories = sorted({r['error_category'] for r in rows})
    output_rows = []
    for model in ['All', 'Granite', 'Qwen']:
        for change in ['All', 'corrected', 'introduced_error']:
            subset = [r for r in rows if (model == 'All' or model_name(r['model']) == model)
                      and (change == 'All' or r['change'] == change)]
            denom = len(subset)
            counts = Counter(r['error_category'] for r in subset)
            for category in categories:
                output_rows.append({
                    'model': model,
                    'change': change,
                    'error_category': category,
                    'count': counts[category],
                    'subset_n': denom,
                    'percent_within_subset': f'{100 * counts[category] / denom:.2f}' if denom else ''
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    all_categories = Counter(r['error_category'] for r in rows)
    lines = [
        '# Manual error analysis — approved review cases',
        '',
        'Source: `result/Arabic_Test/manual_error_review.csv`, combined from the Munira and Maryam approved Excel files.',
        '',
        '| Measure | Count |',
        '|---|---:|',
        f'| Reviewed cases | {len(rows)} |',
        f'| Munira | {reviewers["Munira"]} |',
        f'| Maryam | {reviewers["Maryam"]} |',
        f'| Granite | {models["Granite"]} |',
        f'| Qwen | {models["Qwen"]} |',
        f'| Corrected by P6 | {changes["corrected"]} |',
        f'| New errors under P6 | {changes["introduced_error"]} |',
        '',
        '## Assigned error categories',
        '',
        '| Category | Count | Percent of selected cases |',
        '|---|---:|---:|'
    ]
    for category in categories:
        n = all_categories[category]
        lines.append(f'| {category} | {n} | {100 * n / len(rows):.1f}% |')
    lines += [
        '',
        '## Interpretation limits',
        '',
        '- The 50 cases were purposefully sampled to balance corrected errors against introduced errors; the category proportions are **not** population-wide error rates.',
        '- Error categories describe observed answer patterns and reviewer interpretations. They cannot establish the models\' hidden internal reasoning.',
        '- `baseline_probability` and `proposed_probability` are normalized True/False scores, not necessarily calibrated probabilities.',
        '- The independent 1,000-item dataset label review is documented separately in `annotation_review/`; its observed agreement is 98.4% and Cohen\'s kappa is 0.9680, reproduced with `python analyze_agreement.py`.',
        '- The approved source cases are backed up in `manual_review/approved_cases.csv`; running the inference pipeline again may regenerate unreviewed sheets. Keep originals and reruns separate.',
        '',
        'Reproduce summary: `python analyze_manual_errors.py`.',
        ''
    ]
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    args.notes.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Validated {len(rows)} approved cases; generated {args.output} and {args.notes}')
    print('Category totals:', dict(all_categories))


if __name__ == '__main__':
    main()
