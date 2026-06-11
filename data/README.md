# Dataset download instructions

Place the five CSV files in this directory with the exact names below.

| File | Source |
|------|--------|
| `framingham.csv` | Framingham Heart Study (Kaggle: "Framingham heart study dataset") |
| `cleveland.csv` | UCI Heart Disease, `processed.cleveland.data` converted to CSV |
| `hungarian.csv` | UCI Heart Disease, `processed.hungarian.data` converted to CSV |
| `switzerland.csv` | UCI Heart Disease, `processed.switzerland.data` converted to CSV |
| `long_beach_va.csv` | UCI Heart Disease, `processed.va.data` converted to CSV |

UCI Heart Disease repository: https://archive.ics.uci.edu/dataset/45/heart+disease

## Converting the UCI `.data` files

The four UCI files share the 13-attribute processed schema. Convert with:

```python
import pandas as pd
cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","num"]
for src, dst in [("processed.cleveland.data", "cleveland.csv"),
                 ("processed.hungarian.data", "hungarian.csv"),
                 ("processed.switzerland.data", "switzerland.csv"),
                 ("processed.va.data", "long_beach_va.csv")]:
    pd.read_csv(src, names=cols, na_values="?").to_csv(dst, index=False)
```

Missing values (`?` in the UCI files) are handled by the preprocessing
pipeline via site-specific median imputation computed on the training
partition only.
