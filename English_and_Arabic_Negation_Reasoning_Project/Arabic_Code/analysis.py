from collections import defaultdict
import csv
import json
import math
from pathlib import Path

FIELDS = ('item_id', 'group_id', 'pair_id', 'partition', 'construction', 'polarity', 'statement_ar', 'label')
ORIGINS = {'N': 'arabic_authored_draft', 'T': 'translated_from_english'}


def read_dataset(path, partition, size):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f'Unexpected columns: {path}')
        rows = list(reader)
    if len(rows) != size or len({r['item_id'] for r in rows}) != size:
        raise ValueError(f'Expected {size} unique items: {path}')
    groups, pairs = defaultdict(list), defaultdict(list)
    for row in rows:
        if any(not row[k].strip() for k in FIELDS):
            raise ValueError(f'Missing field: {row}')
        if row['partition'] != partition or row['label'].lower() not in ('true', 'false') or row['polarity'] not in ('affirmative', 'negated'):
            raise ValueError(f'Invalid row: {row}')
        if row['item_id'][0] not in ORIGINS:
            raise ValueError(f'Unknown source prefix: {row["item_id"]}')
        row['label'] = row['label'].lower() == 'true'
        row['origin'] = ORIGINS[row['item_id'][0]]
        row['sentence'] = row['statement_ar']
        groups[row['group_id']].append(row)
        pairs[row['pair_id']].append(row)
    if any(len(v) != 4 or len({r['origin'] for r in v}) != 1 or sum(r['label'] for r in v) != 2 for v in groups.values()):
        raise ValueError('Invalid four-item group')
    if any(len(v) != 2 or len({r['group_id'] for r in v}) != 1 or {r['polarity'] for r in v} != {'affirmative', 'negated'} or {r['label'] for r in v} != {True, False} for v in pairs.values()):
        raise ValueError('Invalid matched pair')
    if len({r['statement_ar'].strip() for r in rows}) != size:
        raise ValueError('Duplicate Arabic statement')
    return rows


def load_prompts(path):
    prompts = json.loads(Path(path).read_text(encoding='utf-8'))['prompts']
    if [p['id'] for p in prompts] != [f'P{i}' for i in range(1, 8)]:
        raise ValueError('Expected P1 through P7 in order')
    for prompt in prompts:
        template = prompt['template']
        remaining = template.replace('{statement_ar}', '')
        if template.count('{statement_ar}') != 1 or '{' in remaining or '}' in remaining or not prompt['name'].strip():
            raise ValueError(f'Invalid template: {prompt["id"]}')
        if 'True' not in template or 'False' not in template:
            raise ValueError(f'Missing answer labels: {prompt["id"]}')
    return prompts


def read_predictions(path, source):
    with Path(path).open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != len(source):
        raise ValueError(f'Incomplete predictions: {path}')
    for expected, actual in zip(source, rows):
        if any(actual.get(k) != v for k, v in expected.items()) or type(actual.get('label')) is not bool:
            raise ValueError(f'Prediction/source mismatch: {expected["item_id"]}')
        p = actual.get('p_true_normalized')
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 or type(actual.get('prediction')) is not bool or actual['prediction'] != (p > 0.5):
            raise ValueError(f'Invalid binary score: {expected["item_id"]}')
    return rows


def count_correct(rows):
    return sum(r['label'] == r['prediction'] for r in rows)


def percentage(numerator, denominator):
    if denominator <= 0:
        raise ValueError('Empty metric subgroup')
    return 100 * numerator / denominator


def pair_rows(rows):
    pairs = defaultdict(list)
    for r in rows:
        pairs[r['pair_id']].append(r)
    return [{'pair_id': key, 'group_id': v[0]['group_id'], 'origin': v[0]['origin'], 'both_correct': count_correct(v) == 2} for key, v in sorted(pairs.items())]


def metrics(rows):
    subgroups = {'overall': rows}
    for key in ('polarity', 'origin'):
        for value in sorted({r[key] for r in rows}):
            subgroups[f'{key}:{value}'] = [r for r in rows if r[key] == value]
    for value in sorted({r['construction'] for r in rows}):
        subgroups[f'construction:{value}'] = [r for r in rows if r['construction'] == value]
        subset = [r for r in rows if r['construction'] == value and r['polarity'] == 'negated']
        if subset:
            subgroups[f'negated_construction:{value}'] = subset
    result = {name: {'n': len(v), 'correct': count_correct(v), 'accuracy_pct': round(percentage(count_correct(v), len(v)), 8)} for name, v in subgroups.items()}
    pairs = pair_rows(rows)
    correct = sum(p['both_correct'] for p in pairs)
    result['matched_pair'] = {'n': len(pairs), 'correct': correct, 'accuracy_pct': round(percentage(correct, len(pairs)), 8)}
    return result


def paired_change(reference, candidate):
    if [r['item_id'] for r in candidate] != [r['item_id'] for r in reference]:
        raise ValueError('Unaligned paired comparison')
    counts = {'both_correct': 0, 'both_wrong': 0, 'corrected': 0, 'introduced_error': 0}
    for a, b in zip(reference, candidate):
        correct_a = a['label'] == a['prediction']
        correct_b = b['label'] == b['prediction']
        key = ('both_correct' if correct_b else 'introduced_error') if correct_a else ('corrected' if correct_b else 'both_wrong')
        counts[key] += 1
    return {'n': len(reference), **counts, 'gain_pp': round(percentage(counts['corrected'] - counts['introduced_error'], len(reference)), 8)}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows: {path}')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, source, models, prompts):
    output = Path(output)
    summaries, breakdowns, changes, predictions, pairs = [], [], [], [], []
    for model, model_id in models.items():
        all_runs = {}
        for p in prompts:
            pid = p['id']
            folder = output / model / pid
            if not (folder / 'completion.json').is_file():
                raise ValueError(f'Incomplete run: {folder}')
            all_runs[pid] = read_predictions(folder / 'predictions.jsonl', source)
        for p in prompts:
            pid = p['id']
            rows = all_runs[pid]
            score = metrics(rows)
            baseline = paired_change(all_runs['P1'], rows)
            ids = {'model': model_id, 'prompt_id': pid, 'prompt_name': p['name']}
            summaries.append({**ids, 'n': len(rows), 'overall_accuracy_pct': score['overall']['accuracy_pct'], 'affirmative_accuracy_pct': score['polarity:affirmative']['accuracy_pct'], 'negated_accuracy_pct': score['polarity:negated']['accuracy_pct'], 'matched_pair_n': score['matched_pair']['n'], 'matched_pair_accuracy_pct': score['matched_pair']['accuracy_pct'], 'translated_accuracy_pct': score['origin:translated_from_english']['accuracy_pct'], 'arabic_authored_accuracy_pct': score['origin:arabic_authored_draft']['accuracy_pct'], 'gain_vs_baseline_pp': baseline['gain_pp'], 'corrected_vs_baseline': baseline['corrected'], 'introduced_errors_vs_baseline': baseline['introduced_error']})
            for name, s in score.items():
                breakdowns.append({**ids, 'metric': name, **s})
            changes.append({**ids, 'reference_prompt': 'P1', **baseline})
            if pid == 'P6':
                changes.append({**ids, 'reference_prompt': 'P7', **paired_change(all_runs['P7'], rows)})
            for r in rows:
                predictions.append({**ids, **{k: r[k] for k in FIELDS}, 'origin': r['origin'], 'prediction': r['prediction'], 'p_true_normalized': r['p_true_normalized'], 'correct': r['label'] == r['prediction']})
            for pair in pair_rows(rows):
                pairs.append({**ids, **pair})
    write_csv(output / 'development_summary.csv', summaries)
    write_csv(output / 'development_breakdown.csv', breakdowns)
    write_csv(output / 'paired_comparisons.csv', changes)
    write_csv(output / 'development_predictions.csv', predictions)
    write_csv(output / 'matched_pairs.csv', pairs)
    return summaries
