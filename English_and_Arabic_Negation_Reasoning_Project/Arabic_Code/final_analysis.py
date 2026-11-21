from collections import defaultdict
import random
import numpy as np
from analysis import FIELDS, metrics, pair_rows, paired_change, write_csv


def cluster_comparison(reference, candidate, metric):
    if [r['item_id'] for r in reference] != [r['item_id'] for r in candidate]:
        raise ValueError('Unaligned comparison')
    groups = defaultdict(lambda: [0, 0, 0])
    if metric == 'matched_pair':
        a, b = pair_rows(reference), pair_rows(candidate)
        for x, y in zip(a, b):
            if x['pair_id'] != y['pair_id']:
                raise ValueError('Unaligned pairs')
            v = groups[x['group_id']]
            v[0] += int(x['both_correct'])
            v[1] += int(y['both_correct'])
            v[2] += 1
    else:
        for x, y in zip(reference, candidate):
            if metric == 'negated' and x['polarity'] != 'negated':
                continue
            v = groups[x['group_id']]
            v[0] += int(x['label'] == x['prediction'])
            v[1] += int(y['label'] == y['prediction'])
            v[2] += 1
    values = np.array([groups[k] for k in sorted(groups)], dtype=np.int64)
    a, b, n = values.sum(axis=0)
    rng = np.random.default_rng(42)
    sample = values[rng.integers(0, len(values), size=(5000, len(values)))].sum(axis=1)
    gains = 100 * (sample[:, 1] - sample[:, 0]) / sample[:, 2]
    low, high = np.quantile(gains, [.025, .975])
    return {'metric':metric, 'n':int(n), 'reference_accuracy_pct':round(100*float(a)/n,8), 'candidate_accuracy_pct':round(100*float(b)/n,8), 'gain_pp':round(100*float(b-a)/n,8), 'ci95_low_pp':round(float(low),4), 'ci95_high_pp':round(float(high),4), 'bootstrap_groups':len(values), 'bootstrap_replicates':5000}


def report(output, models, prompts, runs, comparisons, prefix):
    summaries, breakdown, preds, pairs, intervals, changes = [], [], [], [], [], []
    for model, model_id in models.items():
        for p in prompts:
            pid = p['id']
            rows = runs[model][pid]
            score = metrics(rows)
            baseline = paired_change(runs[model]['P1'], rows)
            ids = {'model':model_id, 'prompt_id':pid, 'prompt_name':p['name']}
            summaries.append({**ids, 'n':len(rows), 'overall_accuracy_pct':score['overall']['accuracy_pct'], 'affirmative_accuracy_pct':score['polarity:affirmative']['accuracy_pct'], 'negated_accuracy_pct':score['polarity:negated']['accuracy_pct'], 'matched_pair_n':score['matched_pair']['n'], 'matched_pair_accuracy_pct':score['matched_pair']['accuracy_pct'], 'translated_accuracy_pct':score['origin:translated_from_english']['accuracy_pct'], 'arabic_authored_accuracy_pct':score['origin:arabic_authored_draft']['accuracy_pct'], 'predicted_true_pct':100*sum(r['prediction'] for r in rows)/len(rows), 'gain_vs_baseline_pp':baseline['gain_pp']})
            breakdown.extend({**ids,'metric':k,**v} for k,v in score.items())
            preds.extend({**ids,**{k:r[k] for k in FIELDS},'origin':r['origin'],'prediction':r['prediction'],'p_true_normalized':r['p_true_normalized'],'correct':r['label']==r['prediction']} for r in rows)
            pairs.extend({**ids,**pair} for pair in pair_rows(rows))
        for candidate, reference, purpose in comparisons:
            ids = {'model':model_id,'candidate_prompt':candidate,'reference_prompt':reference,'comparison':purpose}
            a,b = runs[model][reference],runs[model][candidate]
            changes.append({**ids,**paired_change(a,b)})
            intervals.extend({**ids,**cluster_comparison(a,b,m)} for m in ('overall','negated','matched_pair'))
    names = {f'{prefix}_summary.csv':summaries,f'{prefix}_breakdown.csv':breakdown,f'{prefix}_predictions.csv':preds,f'{prefix}_matched_pairs.csv':pairs,'paired_comparisons.csv':changes,'confidence_intervals.csv':intervals}
    for name, rows in names.items():
        write_csv(output/name,rows)
    return list(names), summaries


def error_review(output, models, runs, target=50):
    rng = random.Random(42)
    strata = {}
    for model, model_id in models.items():
        for a,b in zip(runs[model]['P1'],runs[model]['P6']):
            ca,cb = a['prediction']==a['label'],b['prediction']==b['label']
            if ca and cb:
                continue
            status = 'corrected' if cb else ('introduced_error' if ca else 'both_wrong')
            record = {'model':model_id,**{k:b[k] for k in FIELDS},'origin':b['origin'],'baseline_prediction':a['prediction'],'baseline_probability':a['p_true_normalized'],'proposed_prediction':b['prediction'],'proposed_probability':b['p_true_normalized'],'change':status,'reviewer':'','error_category':'','explanation':''}
            strata.setdefault((model,status),[]).append(record)
    for key in sorted(strata):
        strata[key].sort(key=lambda r:r['item_id'])
        rng.shuffle(strata[key])
    selected, seen = [], set()
    def take(records, limit):
        count=0
        for r in records:
            if count>=limit or len(selected)>=target:
                break
            if r['item_id'] not in seen:
                seen.add(r['item_id'])
                selected.append(r)
                count+=1
    keys=[(m,s) for m in models for s in ('corrected','introduced_error')]
    for i,key in enumerate(keys):
        take(strata.get(key,[]),target//len(keys)+(i<target%len(keys)))
    for status in ('corrected','introduced_error','both_wrong'):
        for m in models:
            take(strata.get((m,status),[]),target-len(selected))
    name='manual_error_review.csv'
    if selected:
        write_csv(output/name,selected)
    else:
        (output/name).write_text('model,item_id,reviewer,error_category,explanation\n',encoding='utf-8-sig')
    counts=[{'model':models[m],'change':s,'eligible_cases':len(strata.get((m,s),[])),'selected_unique_items':sum(r['model']==models[m] and r['change']==s for r in selected)} for m in models for s in ('corrected','introduced_error','both_wrong')]
    write_csv(output/'error_review_sampling.csv',counts)
    return [name,'error_review_sampling.csv']
