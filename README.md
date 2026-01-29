# Rasa merged project (Grades 1–6)

This project merges your Grade 1–3 and Grade 4–6 bots into one Rasa project.

## Folder layout
- `data/grade_1_3/` : NLU / stories / rules from grades 1–3
- `data/grade_4_6/` : NLU / stories / rules from grades 4–6
- `domain.yml` : merged domain (intents/entities/slots/responses/actions)
- `actions/` : Python actions package
  - `grade_1_3_actions.py`
  - `grade_4_6_actions.py`
  - `__init__.py` imports both modules

> NOTE: You only need ONE `actions/__init__.py` (a package can’t have two).
> We merged by importing both grade action modules from the single `__init__.py`.

## Run
In terminal 1:
```bash
rasa train
rasa run actions
```

In terminal 2:
```bash
rasa shell
```

## If training fails
Run:
```bash
rasa data validate
```

Common fixes:
- duplicate intent names are OK, but if two intents mean different things, the model may confuse them.
- if you have two different custom actions with the SAME `name()` string, rename one of them.

## Added Grades 7–10 (latest upload)

- Training data placed in `data/grade_7_10/` (`nlu.yml`, `stories.yml`, `rules.yml`).
- Domain was merged into the root `domain.yml` and the original was saved as `domain_grade_7_10.yml`.
- Actions saved as `actions/grade_7_10_actions.py` **for reference**.

### Note about duplicate actions
Your uploaded `actions.py` for Grades 7–10 is **identical** to the existing `actions/grade_4_6_actions.py`
(same action names like `action_handle_measurements`).  
To avoid duplicate action registration/import conflicts, the grade_7_10 actions module is **not imported** in
`actions/__init__.py`. If you later change Grade 7–10 actions to be different, tell me and we’ll rename them safely.

