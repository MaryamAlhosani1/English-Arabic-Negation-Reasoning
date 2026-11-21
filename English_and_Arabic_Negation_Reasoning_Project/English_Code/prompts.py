

PROMPTS = {
    "baseline": "",
    "wp1": "Pay attention to any negation and distractors (sentences that don't make sense).",
    "wp2": "Pay attention to any negation (words like not, no, never, etc.).",
    "pp1": "You are a linguistic expert who understands the effect of negation words.",
    "pp2": "You are an expert in analysing negations in sentences.",
}

def build_prompt(statement, condition, chat=True):
    baseline = "Is the following statement True or False?"
    if chat:
        baseline += " Answer only True or False."
    instruction = PROMPTS[condition]
    return " ".join(x for x in (instruction, baseline, statement.strip()) if x)
