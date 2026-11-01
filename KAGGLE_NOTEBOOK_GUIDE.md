# CleanWise on Kaggle

## Cell 1 — extract/upload
Upload `cleanwise_project_complete.zip` as an input and extract it into `/kaggle/working`.

## Cell 2 — install
```python
%cd /kaggle/working/cleanwise
!pip install -e .
```

## Cell 3 — tests
```python
!pytest -q
```

## Cell 4 — fast end-to-end run
```python
!python -m cleanwise.benchmark --quick --levels 0.05
!python -m cleanwise.train --model ridge
!python -m cleanwise.report
```

## Cell 5 — inspect
```python
import json, pandas as pd
print(json.load(open("artifacts/evaluation_metrics.json")))
print(json.load(open("artifacts/baseline_metrics.json")))
display(pd.read_csv("artifacts/heldout_predictions.csv").head())
```

## Cell 6 — larger experiment
```python
!python -m cleanwise.benchmark --seed 42 --repeats 2 --levels 0.05 0.15 0.30
!python -m cleanwise.train --model random_forest
!python -m cleanwise.report
```

The four built-in datasets are for software validation. Add a much larger independent dataset
collection before treating the output as research evidence.
