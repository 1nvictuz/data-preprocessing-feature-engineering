# Data Preprocessing and Feature Engineering

AI Developer Boot Camp final assignment 1 of 5.

## Dataset
Download the course-provided `example_dataset.csv` from the assignment's Google Drive link and place it in this directory. **Do not publish the raw dataset unless the license/permission explicitly allows it.** Record its provider, link, date accessed, and applicable usage terms before uploading it publicly. The script does not download data itself.

## Install and run (Windows PowerShell)

```powershell
cd C:\AI-Developer\DataPreprocessing
py -V:3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preprocess.py
```

If the program cannot identify your label column, set `TARGET_COLUMN` near the top of `preprocess.py` to the correct CSV header.

## Method
- Checks missing values, duplicates, empty labels, rare labels, and labels differing only by case. Review flagged labels manually; the script cannot know if a label is factually correct.
- Removes exact duplicate rows and rows with missing labels, then splits 80% train / 20% evaluation with random seed **42** (stratified where feasible).
- Drops common ID columns, creates a numeric interaction if two numeric features exist, and fits median imputation, standard scaling, and categorical imputation/one-hot encoding using **training data only**.
- Reuses the fitted preprocessing transforms on evaluation data. Saves both splits in `output/` with the labels.
- Fits a **majority-class baseline** on training labels, evaluates accuracy on the held-out evaluation set, and stores an audit and package versions in `output/report.json`.

## Results
Run the script and copy the actual counts, flagged label issues, baseline accuracy, and versions from `output/report.json` into your submission. Do **not** invent results before running.

## Limitations
The dataset's target column and licensing must be verified. Rare categories are flagged for review, not automatically rewritten. Small or imbalanced datasets produce unstable accuracy estimates. Data leakage can still arise from repeated people or entities across records; consider group-based or time-based splits if appropriate. The interaction term is educational and may not improve predictive performance.

## GitHub
Upload `preprocess.py`, `requirements.txt`, `README.md`, and (if appropriate) anonymized output reports. Do **not** upload `.venv/`, credentials, personal information, or an unlicensed CSV.
