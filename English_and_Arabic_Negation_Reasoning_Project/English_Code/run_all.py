"""Single-GPU runner for the ten English partial-reproduction conditions.
Uses the upstream run.py, dataset.py, load_model.py and evaluate.py.
No GitHub credentials, uploads, training or Arabic-test evaluation.
"""
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_FLAX"] = "0"

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SAMPLE = PROJECT / 'English_Data/english_test_1100.jsonl'
PROMPT_IDS = ['baseline', 'wp1', 'wp2', 'pp1', 'pp2']
MODELS = {'granite': 'ibm-granite/granite-3.0-2b-instruct', 'qwen': 'Qwen/Qwen1.5-1.8B-Chat'}
PUBLISHED = {'granite': {'all': [57.77, 60.94, 60.29, 60.15, 59.21], 'distractor_negation': [7.22, 24.29, 32.36, 16.74, 18.12]}, 'qwen': {'all': [53.61, 53.01, 50.01, 51.94, 50.66], 'distractor_negation': [4.37, 5.96, 16.55, 10.93, 10.82]}}
SOURCE_KEYS = ('pattern_id', 'pattern', 'test_id', 'negation_type', 'semantic_type', 'syntactic_scope', 'isDistractor', 'label', 'sentence', 'source_row')
CODE_FILES = ['run_all.py', 'run.py', 'load_model.py', 'config.py', 'dataset.py', 'evaluate.py', 'fewshot.py', 'optimizer.py', 'prompts.py', '../requirements.txt', '../run_study.py', '../experiment_settings.json', 'configs/granite.yaml', 'configs/qwen.yaml']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(path)


def read_rows(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_sample():
    rows = read_rows(SAMPLE)
    meta = read_json(PROJECT / 'English_Data/sample_manifest.json')
    if sha(SAMPLE) != meta['sample_sha256']:
        raise ValueError('Frozen sample checksum mismatch')
    if len(rows) != 1100 or len({r['source_row'] for r in rows}) != 1100:
        raise ValueError('Expected 1100 unique source rows')
    if Counter(r['pattern_id'] for r in rows) != {k: 100 for k in range(1,12)}:
        raise ValueError('Expected 100 statements per pattern')
    if any(type(r['label']) is not bool or type(r['isDistractor']) is not bool or not r['sentence'].strip() for r in rows):
        raise ValueError('Malformed sample')
    return rows


def validate_run(run_dir, source_rows):
    """Re-check all predictions and key accuracies from raw output, not just row count."""
    run_dir = Path(run_dir)
    preds = read_rows(run_dir / 'test.jsonl')
    if len(preds) != len(source_rows):
        raise ValueError(f'Wrong prediction count in {run_dir}')
    for src, pred in zip(source_rows, preds):
        if any(src[k] != pred.get(k) for k in SOURCE_KEYS):
            raise ValueError(f'Source/prediction mismatch at row {src["source_row"]}')
        p = pred.get('p_true_normalized')
        if not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1 or type(pred.get('prediction')) is not bool or pred['prediction'] != (p > .5):
            raise ValueError(f'Invalid probability or prediction at row {src["source_row"]}')
    flags=(
        ('all', lambda r: True),
        ('distractor_negation', lambda r: r['isDistractor'] and r['negation_type'] != 'affirmation'),
        ('all_negations', lambda r: r['negation_type'] != 'affirmation'),
        ('all_affirmations', lambda r: r['negation_type'] == 'affirmation'),
        ('distractor_affirmation', lambda r: r['isDistractor'] and r['negation_type'] == 'affirmation'),
        ('input_negation', lambda r: not r['isDistractor'] and r['negation_type'] != 'affirmation'),
        ('input_affirmation', lambda r: not r['isDistractor'] and r['negation_type'] == 'affirmation'),
    )
    results = read_json(run_dir / 'test_results.json')
    for metric, keep in flags:
        sub = [r for r in preds if keep(r)]
        if not sub:
            raise ValueError(f'Empty subgroup: {metric}')
        expected = dict(TP=0, TN=0, FP=0, FN=0)
        for r in sub:
            expected[('T' if r['label']==r['prediction'] else 'F')+('P' if r['prediction'] else 'N')] += 1
        score = round(((expected['TP']+expected['TN'])/len(sub))*100,2)
        got = results[metric]
        if any(got.get(k)!=v for k,v in expected.items()) or abs(got['accuracy']-score)>1e-7:
            raise ValueError(f'Scorer mismatch for {metric}: {run_dir}')
    return results


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def summarize(output, source_rows):
    rows=[]
    for name in MODELS:
        base = None
        for i,p in enumerate(PROMPT_IDS):
            run=output/name/p
            if not (run/'completion.json').is_file():
                raise ValueError(f'Incomplete run: {name}/{p}')
            r=validate_run(run, source_rows)
            if i==0: base=r
            result={'model': MODELS[name], 'prompt': p.upper(), 'n':len(source_rows), 'distractor_negation_n':sum(r['distractor_negation'][k] for k in ('TP','TN','FP','FN'))}
            for metric,prefix in [('all','overall'),('distractor_negation','distractor_negation')]:
                published=PUBLISHED[name][metric][i]; reproduced=r[metric]['accuracy']; baseline=base[metric]['accuracy']; pub_base=PUBLISHED[name][metric][0]
                result.update({f'{prefix}_published_pct':published,f'{prefix}_reproduced_pct':reproduced,f'{prefix}_difference_pp':round(reproduced-published,2),f'{prefix}_published_gain_pp':round(published-pub_base,2),f'{prefix}_reproduced_gain_pp':round(reproduced-baseline,2)})
            result.update(affirmative_accuracy_pct=r['all_affirmations']['accuracy'],negated_accuracy_pct=r['all_negations']['accuracy'],input_negation_accuracy_pct=r['input_negation']['accuracy'],distractor_affirmation_accuracy_pct=r['distractor_affirmation']['accuracy'])
            rows.append(result)
    csv_write(output/'english_comparison.csv',rows)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,help='New session directory, or existing session with --resume')
    ap.add_argument('--resume',action='store_true',help='Resume only validated runs in exactly the same recorded session')
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--check-data',action='store_true',help='Offline verification only; no GPU or model downloads')
    args=ap.parse_args()
    source_rows=validate_sample()
    if args.check_data:
        print('Verified 1100 original test statements; 100 per pattern. No model inference performed.')
        return
    if args.batch_size<1: ap.error('--batch-size must be positive')
    if args.resume and args.output_dir is None: ap.error('--resume requires --output-dir')
    import torch
    import yaml
    if not torch.cuda.is_available(): raise RuntimeError('No GPU detected. Select GPU runtime in Colab.')
    if torch.cuda.device_count()!=1: raise RuntimeError('Use one visible GPU for this reproduction.')
    cap=torch.cuda.get_device_capability(0)
    dtype='bfloat16' if cap[0]>=8 and torch.cuda.is_bf16_supported() else 'float16'
    output=(args.output_dir or PROJECT/'results'/dt.datetime.now(dt.timezone.utc).strftime('english_%Y%m%dT%H%M%S_%fZ')).expanduser().resolve()
    manifest_path=output/'session.json'
    versions={k:md.version(k) for k in ('torch','transformers','accelerate','bitsandbytes','datasets','pandas','numpy','huggingface-hub','PyYAML','wandb','peft','sentencepiece')}
    base_manifest={'study':'English partial reproduction; 1100 statements, 100 per pattern, not full test set','upstream_commit':read_json(PROJECT/'English_Data/sample_manifest.json')['upstream_commit'],'sample_sha256':sha(SAMPLE),'code_sha256':{f:sha(ROOT/f) for f in CODE_FILES},'models':MODELS,'prompt_ids':PROMPT_IDS,'seed':42,'gpu':torch.cuda.get_device_name(0),'gpu_capability':list(cap),'cuda':torch.version.cuda,'torch_dtype':dtype,'quantization':4,'initial_batch_size':args.batch_size,'python':platform.python_version(),'packages':versions}
    if args.resume:
        m=read_json(manifest_path)
        if any(m.get(k)!=v for k,v in base_manifest.items()): raise ValueError('Session manifest differs. Start a fresh output directory; no stale results will be reused.')
        revisions=m['model_revisions']
    else:
        if output.exists() and any(output.iterdir()): raise FileExistsError('Output is not empty. Choose a fresh directory or use --resume.')
        output.mkdir(parents=True,exist_ok=True)
        revisions=read_json(PROJECT / "experiment_settings.json")["model_revisions"]
        if any(not re.fullmatch(r'[0-9a-f]{40}',v or '') for v in revisions.values()): raise RuntimeError('Cannot resolve exact model revision')
        write_json(manifest_path,{**base_manifest,'model_revisions':revisions,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    print(f'GPU: {base_manifest["gpu"]} | compute dtype: {dtype} | results: {output}',flush=True)
    for model in MODELS:
        for prompt in PROMPT_IDS:
            run_dir=output/model/prompt
            cfg=yaml.safe_load((ROOT/'configs'/f'{model}.yaml').read_text())
            cfg.update(model_revision=revisions[model],output_dir=str(run_dir),torch_dtype=dtype,per_device_eval_batch_size=args.batch_size,sample_file=str(SAMPLE),prompt_type=prompt)
            cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
            done=run_dir/'completion.json'
            if done.exists():
                record=read_json(done)
                if record.get('config_sha256')!=cfg_hash or any(sha(run_dir/f)!=record['files_sha256'][f] for f in ('test.jsonl','test_results.json','test.preds','config.yaml','run.log')):
                    raise ValueError(f'Existing outputs failed manifest check: {run_dir}')
                validate_run(run_dir,source_rows)
                print(f'Verified completed run: {model}/{prompt}',flush=True); continue
            if run_dir.exists():

                archive=output/'incomplete'/model/(prompt+'_'+str(time.time_ns()))
                archive.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(run_dir),str(archive))
            run_dir.mkdir(parents=True)
            (run_dir/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
            start=time.monotonic()
            with (run_dir/'run.log').open('w',encoding='utf8') as log:
                process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py'),'--config',str(run_dir/'config.yaml')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env={**os.environ,'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'})
                for line in process.stdout:
                    print(line,end='',flush=True); log.write(line); log.flush()
                code=process.wait()
            if code!=0: raise RuntimeError(f'{model}/{prompt} failed ({code}). See {run_dir}/run.log; fix issue and resume or start fresh.')
            r=validate_run(run_dir,source_rows)
            write_json(done,{'config_sha256':cfg_hash,'files_sha256':{f:sha(run_dir/f) for f in ('test.jsonl','test_results.json','test.preds','config.yaml','run.log')},'seconds':round(time.monotonic()-start,3),'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
            print(f'{model}/{prompt}: overall {r["all"]["accuracy"]:.2f}% | distractor negation {r["distractor_negation"]["accuracy"]:.2f}% | {(time.monotonic()-start)/60:.1f} min',flush=True)
    rows=summarize(output,source_rows)
    print('\nEnglish partial reproduction results:')
    for r in rows: print(f'{r["model"]:40s} {r["prompt"]:10s} overall={r["overall_reproduced_pct"]:6.2f}% distractor={r["distractor_negation_reproduced_pct"]:6.2f}%')
    print('Results saved:',output/'english_comparison.csv')

if __name__=='__main__': main()
