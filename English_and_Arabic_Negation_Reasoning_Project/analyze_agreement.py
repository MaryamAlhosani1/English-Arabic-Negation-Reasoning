import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'annotation_review'
DATA = ROOT / 'Arabic_Data'


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def keyed(rows, id_field='item_id'):
    output = {}
    for row in rows:
        key = row[id_field].strip()
        if not key or key in output:
            raise ValueError(f'Missing or duplicate item ID: {key}')
        output[key] = row
    return output


def check_label(value, name):
    label = str(value).strip().upper()
    if label not in ('TRUE', 'FALSE'):
        raise ValueError(f'Invalid label for {name}: {value!r}')
    return label


def main():
    munira = keyed(load(EVIDENCE / 'labels_munira.csv'))
    maryam = keyed(load(EVIDENCE / 'labels_maryam.csv'))
    adjudication = keyed(load(EVIDENCE / 'adjudication.csv'))
    final = keyed(load(DATA / 'Development.csv') + load(DATA / 'Test.csv'))
    ids = set(final)

    if not (len(final) == len(munira) == len(maryam) == 1000):
        raise ValueError('Expected 1,000 complete dataset and reviewer records')
    if not (ids == set(munira) == set(maryam)):
        raise ValueError('Reviewer IDs do not match dataset IDs')

    counts = Counter()
    disagreed = set()
    for item_id, dataset_row in final.items():
        a = munira[item_id]
        b = maryam[item_id]
        statement = dataset_row['statement_ar'].strip()
        if a['statement_ar'].strip() != statement or b['statement_ar'].strip() != statement:
            raise ValueError(f'Statement mismatch: {item_id}')
        la = check_label(a['label'], f'Munira {item_id}')
        lb = check_label(b['label'], f'Maryam {item_id}')
        lf = check_label(dataset_row['label'], f'Final {item_id}')
        counts[la, lb] += 1
        if la != lb:
            disagreed.add(item_id)
        elif lf != la:
            raise ValueError(f'The released label disagrees with both reviewers: {item_id}')

    if disagreed != set(adjudication):
        raise ValueError('Disagreement IDs and adjudication IDs do not match')

    for item_id in disagreed:
        resolution = adjudication[item_id]
        if resolution['statement_ar'].strip() != final[item_id]['statement_ar'].strip():
            raise ValueError(f'Adjudication text mismatch: {item_id}')
        if check_label(resolution['munira_label'], f'Adjudication Munira {item_id}') != check_label(munira[item_id]['label'], item_id):
            raise ValueError(f'Munira adjudication label mismatch: {item_id}')
        if check_label(resolution['maryam_label'], f'Adjudication Maryam {item_id}') != check_label(maryam[item_id]['label'], item_id):
            raise ValueError(f'Maryam adjudication label mismatch: {item_id}')
        if check_label(resolution['resolved_label'], f'Adjudication final {item_id}') != check_label(final[item_id]['label'], item_id):
            raise ValueError(f'Adjudication resolution mismatches final dataset: {item_id}')

    n = len(final)
    same_true = counts['TRUE', 'TRUE']
    same_false = counts['FALSE', 'FALSE']
    munira_true_maryam_false = counts['TRUE', 'FALSE']
    munira_false_maryam_true = counts['FALSE', 'TRUE']
    matches = same_true + same_false
    disagreements = n - matches
    observed = matches / n
    munira_true = (same_true + munira_true_maryam_false) / n
    maryam_true = (same_true + munira_false_maryam_true) / n
    expected = munira_true * maryam_true + (1 - munira_true) * (1 - maryam_true)
    kappa = (observed - expected) / (1 - expected)

    metrics = [
        ('total_items', n),
        ('both_true', same_true),
        ('both_false', same_false),
        ('munira_true_maryam_false', munira_true_maryam_false),
        ('munira_false_maryam_true', munira_false_maryam_true),
        ('agreements', matches),
        ('disagreements_resolved', disagreements),
        ('observed_agreement', f'{observed:.6f}'),
        ('expected_chance_agreement', f'{expected:.6f}'),
        ('cohens_kappa', f'{kappa:.9f}'),
        ('final_dataset_labels_validated', n),
    ]
    with (EVIDENCE / 'agreement_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerows(metrics)

    report = f'''# Arabic dataset human annotation agreement

The dataset review records contain complete, separate annotations by Munira and Maryam. The package includes their separate 1,000-item review workbooks, the agreement workbook, and Suhaila's adjudication record. The script validates every label against the released development and held-out test CSV files.

| Measure | Result |
|---|---:|
| Reviewed items | {n:,} |
| Both reviewers: True | {same_true} |
| Both reviewers: False | {same_false} |
| Munira True / Maryam False | {munira_true_maryam_false} |
| Munira False / Maryam True | {munira_false_maryam_true} |
| Agreements | {matches} |
| Disagreements adjudicated | {disagreements} |
| Observed agreement | {observed:.2%} |
| Expected chance agreement | {expected:.6f} |
| **Cohen's kappa** | **{kappa:.4f}** |

All {n:,} item IDs and statement texts match the study dataset. Every one of the {disagreements} disagreements has a corresponding adjudication record, and all released labels match either the agreed reviewer label or the documented adjudicated label.

**Reproduce the analysis** from the repository root with `python analyze_agreement.py`. The calculation uses the exported reviewer labels stored under `annotation_review/`. Original Excel sources are included alongside those exports.
'''
    (EVIDENCE / 'Agreement_Report.md').write_text(report, encoding='utf-8')
    print(f'Validated {n} items: {matches} agreements, {disagreements} resolved disagreements, kappa={kappa:.9f}')


if __name__ == '__main__':
    main()
