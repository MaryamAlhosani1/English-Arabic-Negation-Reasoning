# English and Arabic Negation Reasoning

## Setup

Use a Python environment with CUDA-enabled PyTorch and one NVIDIA L4 GPU. The recorded software and model revisions are in `experiment_settings.json`. Run from the repository root.

```bash
python -m pip install --only-binary=sentencepiece,numpy,bitsandbytes -r requirements.txt
python check_environment.py
python run_study.py --check
```

## English reproduction

```bash
python run_study.py --stage english
```

The selected English test subset contains 1,100 statements (100 per pattern), evaluated with two original models and five prompt conditions.

## Arabic development

```bash
python run_study.py --stage development
```

Use the 200 development statements to evaluate the original seven conditions, three additional negation-specific formulations and a shorter general control. Rank the candidates by mean overall accuracy across the two models, breaking ties by negated accuracy, matched-pair accuracy and the predefined candidate order. The original development selected P6_V4 and P7_V2. These are frozen for the recorded test.

## Arabic test

```bash
python run_study.py --stage test
```

Evaluate the frozen seven prompts on the separate 800-statement test set. Do not use test results to revise or select prompts. The test produces the accuracy tables, grouped confidence intervals, predictions and a 50-item manual error review sheet. The recorded `result/Arabic_Test/manual_error_review.csv` has subsequently been populated from the team-approved Munira and Maryam review files. Running the pipeline again can regenerate a blank review sheet; use a separate output folder and preserve the published reviewed copy.

## Robustness

```bash
python run_study.py --stage robustness
```

Run after the test. Evaluate three meaning-preserving paraphrases and two reversed prompt orders using the same 800 statements. The main test predictions provide the original-order references.

## Run everything

```bash
python run_study.py --stage all
```

The recorded results are in `result/`. To run without altering the supplied results, use `--output result_2`. A compatible output folder can resume completed conditions. Inference logs and checkpoints are stored locally under `.runs/` and are not part of the public results tables. The development `prompt_selection.json` preserves the recommendation at the development stage; the final approved/frozen conditions for the held-out test are recorded in `Arabic_Code/Arabic_Test_Protocol.json`.


## Manual error review (50 selected test cases)

The recorded review combines the Munira and Maryam review workbooks, with 25 cases per reviewer. The approved Excel sources in `manual_review/`, while the analysis-ready merged data is in `result/Arabic_Test/manual_error_review.csv` (18 columns). This is an evaluation of pre-selected model errors, not new data or an additional inference run.


## Human dataset review and annotation agreement

The completed 1,000-item Arabic dataset review is documented in `annotation_review/`: independent review workbooks for **Munira** and **Maryam**, a separate **Suhaila** adjudication record for the 16 disagreements, the supplied `Agreement.xlsx`, and reproducible CSV exports. All reviewer item IDs, statements, and finalized labels were checked against the released `Arabic_Data/Development.csv` and `Arabic_Data/Test.csv`.

- **Observed agreement:** 984/1,000 = **98.4%**.
- **Cohen's kappa:** **0.9680** (calculated from the 1,000 paired reviewer labels).
- **Adjudication:** all 16 disagreements have a recorded resolution aligned with the final dataset.

```bash
python analyze_agreement.py
```
