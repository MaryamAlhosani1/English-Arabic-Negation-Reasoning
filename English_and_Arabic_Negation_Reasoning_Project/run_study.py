import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['USE_FLAX'] = '0'

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'English_Code'
ARABIC = ROOT / 'Arabic_Code'
sys.path[:0] = [str(ARABIC), str(CODE)]
from run_all import MODELS, PROMPT_IDS, validate_sample, validate_run, read_rows, csv_write, read_json, write_json, sha
from analysis import read_dataset, read_predictions, load_prompts, metrics
from development import run_candidate, report as development_report
from final_analysis import report, error_review

SETTINGS = read_json(ROOT / 'experiment_settings.json')
COMPARISONS = [(f'P{i}', 'P1', 'primary' if i == 6 else 'secondary') for i in range(2, 8)] + [('P6', 'P7', 'primary')] + [('P6', f'P{i}', 'secondary') for i in range(2, 6)]


def inputs():
    for path, digest in {**SETTINGS['data_sha256'], **SETTINGS['prompt_sha256']}.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Input changed: {path}')
    english = validate_sample()
    dev = read_dataset(ROOT / 'Arabic_Data/Development.csv', 'dev', 200)
    test = read_dataset(ROOT / 'Arabic_Data/Test.csv', 'test', 800)
    for k in ('item_id', 'group_id', 'pair_id', 'statement_ar'):
        if {r[k] for r in dev} & {r[k] for r in test}:
            raise ValueError(f'Split overlap: {k}')
    original = load_prompts(ARABIC / 'Arabic_Prompts.json')
    final = load_prompts(ARABIC / 'Arabic_Prompts_Final.json')
    protocol = read_json(ARABIC / 'Development_V2_Protocol.json')
    robust = read_json(ARABIC / 'Arabic_Test_Protocol.json')['robustness_prompts']
    if final[:5] != original[:5] or len(robust) != 5:
        raise ValueError('Frozen study conditions changed')
    return english, dev, test, original, final, protocol, robust


def environment():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Select one CUDA GPU')
    cap = torch.cuda.get_device_capability(0)
    actual = {'gpu': torch.cuda.get_device_name(0), 'gpu_capability': list(cap), 'cuda': torch.version.cuda, 'torch_dtype': 'bfloat16' if cap[0] >= 8 and torch.cuda.is_bf16_supported() else 'float16', 'quantization': 4, 'initial_batch_size': 32, 'seed': 42, 'model_revisions': SETTINGS['model_revisions'], 'packages': {k: metadata.version(k) for k in SETTINGS['packages']}, 'python': sys.version.split()[0]}
    if actual['torch_dtype'] != SETTINGS['torch_dtype']:
        raise RuntimeError('Use an L4 or equivalent BF16 GPU for this replication')
    return actual


def phase_runs(work, rows, prompts, env):
    work.mkdir(parents=True, exist_ok=True)
    sample = work / 'sample.jsonl'
    sample.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return {m: {p['id']: run_candidate(work, m, p, env, rows, sample) for p in prompts} for m in MODELS}


def load_runs(work, rows, prompts):
    return {m: {p['id']: read_predictions(work / m / p['id'] / 'predictions.jsonl', rows) for p in prompts} for m in MODELS}


def english_report(work, out, rows):
    import shutil
    shutil.copyfile(work / 'english_comparison.csv', out / 'english_comparison.csv')
    all_predictions = []
    for m, model in MODELS.items():
        for p in PROMPT_IDS:
            folder = work / m / p
            validate_run(folder, rows)
            all_predictions.extend({'model': model, 'prompt': p.upper(), **r, 'correct': r['label'] == r['prediction']} for r in read_rows(folder / 'test.jsonl'))
    csv_write(out / 'english_predictions.csv', all_predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'english', 'development', 'test', 'robustness'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'result')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    english, dev, test, original, final, protocol, robust = inputs()
    if args.check:
        print('English: 1100; development: 200; test: 800. Input checks passed.')
        return
    out = args.output.resolve()
    if out == ROOT or (out.is_relative_to(ROOT) and out.relative_to(ROOT).parts[0] in ('Results', 'English_Data', 'Arabic_Data', 'English_Code', 'Arabic_Code')) or ROOT.is_relative_to(out):
        raise ValueError('Choose a separate output directory')
    work = out / '.runs'
    if out.exists() and any(out.iterdir()) and not (work / 'session.json').is_file():
        raise FileExistsError('Choose an empty output directory')
    work.mkdir(parents=True, exist_ok=True)
    env = environment()
    code = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*.py')) if 'result' not in p.relative_to(ROOT).parts and not p.is_relative_to(out)}
    settings = {**env, 'data_sha256': SETTINGS['data_sha256'], 'prompt_sha256': SETTINGS['prompt_sha256'], 'code_sha256': code}
    session = work / 'session.json'
    if session.exists():
        if read_json(session) != settings:
            raise ValueError('Run settings changed; choose a new output directory')
    else:
        write_json(session, settings)
    for stage, folder in [('english', 'English'), ('development', 'Arabic_Development'), ('test', 'Arabic_Test'), ('robustness', 'Arabic_Robustness')]:
        if args.stage not in ('all', stage):
            continue
        stage_out = out / folder
        stage_out.mkdir(parents=True, exist_ok=True)
        stage_work = work / folder
        if stage == 'english':
            command = [sys.executable, '-u', str(CODE / 'run_all.py'), '--output-dir', str(stage_work)]
            if (stage_work / 'session.json').exists():
                command.append('--resume')
            subprocess.run(command, cwd=CODE, check=True)
            english_report(stage_work, stage_out, english)
        elif stage == 'development':
            candidates = protocol['candidate_prompts']
            runs = phase_runs(stage_work, dev, original + candidates, env)
            ranking = development_report(stage_out, protocol, original, candidates, runs)
            (stage_out / 'prompt_recommendation.json').replace(stage_out / 'prompt_selection.json')
            print('Development winner:', ranking[0]['prompt_id'], '| Historical frozen test prompt remains unchanged')
        elif stage == 'test':
            runs = phase_runs(stage_work, test, final, env)
            report(stage_out, MODELS, final, runs, COMPARISONS, 'test')
            error_review(stage_out, MODELS, runs)
        else:
            runs = load_runs(work / 'Arabic_Test', test, final)
            extra = phase_runs(stage_work, test, robust, env)
            for m in MODELS:
                runs[m].update(extra[m])
            comparisons = [(p['id'], p['reference'], p['kind']) for p in robust] + [(p['id'], 'P1', 'baseline_gain') for p in robust] + [(p['id'], 'P7', 'control_gain') for p in robust if p['reference'] == 'P6']
            report(stage_out, MODELS, robust, runs, comparisons, 'robustness')
        archive = out / (folder + '_Result.zip')
        temporary = archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in sorted(stage_out.iterdir()):
                if file.is_file():
                    z.write(file, file.name)
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise ValueError('Result archive failed integrity check')
        temporary.replace(archive)
        print('Saved:', archive, flush=True)
    write_json(out / 'environment.json', {**settings, 'last_execution_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
