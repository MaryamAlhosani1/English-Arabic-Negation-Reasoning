import os
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'English_Code'
sys.path.insert(0, str(CODE))
from run_all import MODELS, read_json, write_json, sha as sha256
from analysis import FIELDS, metrics, paired_change, pair_rows, read_predictions, write_csv
RUN_FILES = ('config.yaml', 'predictions.jsonl', 'predictions.preds', 'metrics.json', 'run.log')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run_candidate(output, model, p, old, rows, sample):
    import yaml
    pid = p['id']
    folder = output / model / pid
    cfg = yaml.safe_load((CODE / 'configs' / f'{model}.yaml').read_text())
    cfg.update(model_revision=old['model_revisions'][model], output_dir=str(folder), torch_dtype=old['torch_dtype'], per_device_eval_batch_size=32, sample_file=str(sample), sample_size=len(rows), prompt_type='baseline', arabic_prompt_template=p['template'])
    cfg_hash = digest(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode())
    done = folder / 'completion.json'
    if done.exists():
        record = read_json(done)
        if record['config_sha256'] != cfg_hash or any(sha256(folder / f) != record['files_sha256'][f] for f in RUN_FILES):
            raise ValueError(f'Completed run has changed: {folder}')
        pred = read_predictions(folder / 'predictions.jsonl', rows)
        if metrics(pred) != read_json(folder / 'metrics.json'):
            raise ValueError('Existing candidate metrics mismatch')
        print(f'Verified completed run: {model}/{pid}', flush=True)
        return pred
    if folder.exists():
        backup = output / 'incomplete' / model / f'{pid}_{time.time_ns()}'
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(backup))
    folder.mkdir(parents=True)
    (folder / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    start = time.monotonic()
    with (folder / 'run.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen([sys.executable, '-u', str(CODE / 'run.py'), '--config', str(folder / 'config.yaml')], cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, 'WANDB_MODE':'disabled', 'TOKENIZERS_PARALLELISM':'false'}) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if proc.wait() != 0:
                raise RuntimeError(f'{model}/{pid} failed; see {folder / "run.log"}')
    pred = read_predictions(folder / 'predictions.jsonl', rows)
    scores = metrics(pred)
    write_json(folder / 'metrics.json', scores)
    write_json(done, {'config_sha256':cfg_hash, 'files_sha256':{f:sha256(folder / f) for f in RUN_FILES}, 'seconds':round(time.monotonic()-start, 3), 'completed_utc':datetime.now(timezone.utc).isoformat()})
    print(f'{model}/{pid}: accuracy={scores["overall"]["accuracy_pct"]:.2f}%', flush=True)
    return pred


def cluster_interval(reference, candidate):
    groups = defaultdict(list)
    for a, b in zip(reference, candidate):
        if a['item_id'] != b['item_id']:
            raise ValueError('Unaligned comparison')
        groups[a['group_id']].append(int(b['prediction'] == b['label']) - int(a['prediction'] == a['label']))
    values = [(sum(v), len(v)) for _, v in sorted(groups.items())]
    rng = random.Random(42)
    samples = []
    for _ in range(5000):
        draw = rng.choices(values, k=len(values))
        samples.append(100 * sum(v[0] for v in draw) / sum(v[1] for v in draw))
    samples.sort()
    def quantile(p):
        pos = (len(samples)-1)*p
        lo = int(pos)
        return round(samples[lo]+(pos-lo)*(samples[min(lo+1,len(samples)-1)]-samples[lo]), 4)
    return {'ci95_low_pp':quantile(.025), 'ci95_high_pp':quantile(.975), 'bootstrap_groups':len(values), 'bootstrap_replicates':5000}


def report(output, protocol, original, candidates, runs):
    summary, breakdown, comparisons, predictions, pairs, lengths = [], [], [], [], [], []
    prompts = original + candidates
    for model, model_id in MODELS.items():
        for p in prompts:
            pid = p['id']
            rows = runs[model][pid]
            score = metrics(rows)
            ids = {'model':model_id, 'prompt_id':pid, 'prompt_name':p['name'], 'phase':'V1_archived' if p in original else 'V2_new'}
            baseline = paired_change(runs[model]['P1'], rows)
            summary.append({**ids, 'n':len(rows), 'overall_accuracy_pct':score['overall']['accuracy_pct'], 'affirmative_accuracy_pct':score['polarity:affirmative']['accuracy_pct'], 'negated_accuracy_pct':score['polarity:negated']['accuracy_pct'], 'matched_pair_accuracy_pct':score['matched_pair']['accuracy_pct'], 'translated_accuracy_pct':score['origin:translated_from_english']['accuracy_pct'], 'arabic_authored_accuracy_pct':score['origin:arabic_authored_draft']['accuracy_pct'], 'predicted_true_pct':100*sum(r['prediction'] for r in rows)/len(rows), 'gain_vs_baseline_pp':baseline['gain_pp']})
            for k, v in score.items():
                breakdown.append({**ids, 'metric':k, **v})
            refs = ['P1'] if pid != 'P1' else []
            if pid in protocol['selection']['eligible']:
                refs += [protocol['selection']['control_mapping'][pid]]
                if pid != 'P6':
                    refs += ['P6']
                refs += ['P2', 'P3', 'P4', 'P5']
            for ref in refs:
                prior = runs[model][ref]
                comparisons.append({**ids, 'reference_prompt':ref, **paired_change(prior, rows), **cluster_interval(prior, rows)})
            for r in rows:
                predictions.append({**ids, **{k:r[k] for k in FIELDS}, 'origin':r['origin'], 'prediction':r['prediction'], 'p_true_normalized':r['p_true_normalized'], 'correct':r['label']==r['prediction']})
            pairs.extend({**ids, **pair} for pair in pair_rows(rows))
    for p in prompts:
        prefix = p['template'].removesuffix(original[0]['template']).strip()
        lengths.append({'prompt_id':p['id'], 'instruction_words':len(prefix.split()), 'instruction_characters':len(prefix)})
    ranking=[]
    for priority,pid in enumerate(protocol['selection']['eligible']):
        items=[s for s in summary if s['prompt_id']==pid]
        if len(items)!=len(MODELS):
            raise ValueError('Incomplete candidate')
        ranking.append({'prompt_id':pid, 'mean_overall_accuracy_pct':sum(s['overall_accuracy_pct'] for s in items)/len(items), 'mean_negated_accuracy_pct':sum(s['negated_accuracy_pct'] for s in items)/len(items), 'mean_matched_pair_accuracy_pct':sum(s['matched_pair_accuracy_pct'] for s in items)/len(items), 'tie_priority':priority, 'matched_control':protocol['selection']['control_mapping'][pid]})
    ranking.sort(key=lambda r:(-r['mean_overall_accuracy_pct'], -r['mean_negated_accuracy_pct'], -r['mean_matched_pair_accuracy_pct'], r['tie_priority']))
    for i,row in enumerate(ranking,1):
        row['rank']=i
    selected=ranking[0]
    lookup={p['id']:p for p in prompts}
    proposal=original[:5]+[{**lookup[selected['prompt_id']], 'id':'P6'}, {**lookup[selected['matched_control']], 'id':'P7'}]
    proposal=[{k:p[k] for k in ('id','name','template')} for p in proposal]
    write_json(output/'prompt_recommendation.json', {'status':'Proposed, not frozen; requires team approval', 'selected_candidate':selected['prompt_id'], 'selected_control':selected['matched_control'], 'selection_rule':protocol['selection'], 'prompts':proposal})
    for name,data in [('development_summary.csv',summary),('development_breakdown.csv',breakdown),('paired_comparisons.csv',comparisons),('development_predictions.csv',predictions),('matched_pairs.csv',pairs),('candidate_ranking.csv',ranking),('prompt_lengths.csv',lengths)]:
        write_csv(output/name,data)
    if len(summary)!=22 or len(predictions)!=4400 or len(pairs)!=2200:
        raise ValueError('Unexpected combined report size')
    return ranking
