# Dataset Download Instructions

Place the five CSV files in this directory with the exact filenames listed below before running any experiment script.

## Required files

| Filename | Institution | Source |
|---|---|---|
| `framingham.csv` | Framingham Heart Study | Kaggle |
| `cleveland.csv` | Cleveland Clinic Foundation | UCI Machine Learning Repository |
| `hungarian.csv` | Hungarian Institute of Cardiology | UCI Machine Learning Repository |
| `switzerland.csv` | University Hospital Zurich | UCI Machine Learning Repository |
| `long_beach_va.csv` | Long Beach VA Medical Center | UCI Machine Learning Repository |

## Framingham Heart Study

Download from Kaggle:

```
https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset
```

Rename the downloaded file to `framingham.csv` and place it in this folder.

## UCI Heart Disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA)

Download the four processed files from the UCI Machine Learning Repository:

```
https://archive.ics.uci.edu/dataset/45/heart+disease
```

The archive contains four processed files: `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, and `processed.va.data`. Convert each to CSV using the script below, then place the resulting files in this folder.

### Conversion script

```python
import pandas as pd

COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

MAPPING = {
    "processed.cleveland.data": "cleveland.csv",
    "processed.hungarian.data": "hungarian.csv",
    "processed.switzerland.data": "switzerland.csv",
    "processed.va.data": "long_beach_va.csv",
}

for src, dst in MAPPING.items():
    df = pd.read_csv(src, names=COLS, na_values="?")
    df.to_csv(dst, index=False)
    print(f"Saved {dst}  ({len(df)} rows)")
```

Run this script from the folder where you placed the `.data` files. The resulting CSVs go into `data/`.

## Notes

Missing values (encoded as `?` in the UCI files) are handled automatically by the preprocessing pipeline in `fedcvr/data_utils.py` through site-specific median imputation computed on the training partition only. The Framingham dataset covers a partially overlapping feature set; attributes absent from that source are treated as missing and imputed accordingly.

All datasets are used exclusively for research purposes and are subject to their own respective licenses. Please refer to each source for terms of use.
