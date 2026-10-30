# Data Preprocessing and Feature Engineering

## Overview
This project is part of my AI Developer Boot Camp. I used Python to clean, prepare, and improve a dataset for machine learning.

## What I Did
- Checked for missing values and duplicates.
- Split 100 records into 80 training and 20 evaluation records.
- Applied mean imputation and standard scaling using training data only.
- Created 3 additional features from the original 5.
- Saved the processed datasets as CSV files.

## Tools Used
Python 3.12.10, Pandas 2.3.3, NumPy 2.5.4, Scikit-learn 1.9.1

## How to Run
```bash
pip install -r requirements.txt
python preprocess.py
```

## Results
- Missing values: 0
- Duplicate records: 0
- Total processed features: 8
- Random seed: 42

## Limitations
The course-provided dataset has no target labels, so classification accuracy and a baseline could not be calculated. Dataset redistribution permissions have not been verified.

## What I Learned
I learned how to prepare datasets, create new features, and avoid data leakage by applying preprocessing rules learned only from training data.
