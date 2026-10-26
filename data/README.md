# Data

The IEEE-CIS dataset is licensed and is **not committed**. This directory is
local-only except for this note.

Expected layout after download:

```
data/
├── raw/ieee/
│   ├── train_transaction.csv
│   └── train_identity.csv
└── processed/
    └── ieee.pkl          # generated cache, ~979 MB
```

Download the competition data from Kaggle (accept the rules first):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee
cd data/raw/ieee
unzip ieee-fraud-detection.zip
```

`src/data/ieee.py` joins the two tables, shrinks dtypes and writes the pickle
cache on first use. Both `raw/` and `processed/` are gitignored. The cache is a
pickle because pyarrow currently requires NumPy >= 2 while this project pins
NumPy 1.26.

`models/stream_sample.pkl` is also gitignored: it is a small slice of the
licensed test set used only by the demo dashboard.
