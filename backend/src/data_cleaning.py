"""
data_cleaning.py – reusable functions for cleaning the churn dataset.

Each function has a docstring explaining:
  - WHAT it does
  - WHY it is needed

These functions are called from the notebook AND from train.py so the
cleaning is always identical.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame.

    WHY: We centralise loading here so that the path is always the same
    in the notebook and in train.py.
    """
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {filepath}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING VALUE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame showing missing count and percentage for every column.

    WHY: Before we fix missing values we need to know where they are and
    how many there are.  This table makes it easy to decide the right
    imputation strategy for each column.
    """
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        "missing_count":   missing_count,
        "missing_percent": missing_pct.round(2),
    })
    # Only keep rows where something is actually missing
    report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 3. DROP ROWS WITH MISSING TARGET
# ─────────────────────────────────────────────────────────────────────────────

def drop_missing_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Remove rows where the target column (Churn) is missing.

    WHY: A supervised learning model needs a label for every training row.
    If the target is missing we literally do not know whether that customer
    churned or not, so we cannot use that row to teach the model anything.
    Keeping those rows would corrupt the training signal.
    """
    before = len(df)
    df = df.dropna(subset=[target_col]).copy()
    dropped = before - len(df)
    logger.info(f"Dropped {dropped} rows with missing '{target_col}' (cannot train without a label)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIX DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix incorrect data types:
      - TotalCharges: stored as text → convert to float (some rows are blank strings)
      - tenure:       stored as float → should be integer months
      - SeniorCitizen: stored as 0/1 integer → convert to 'Yes'/'No' string so it
                       is treated as a category by the preprocessor

    WHY:
      TotalCharges – pandas reads it as object because some cells contain a space
      character.  pd.to_numeric(errors="coerce") safely turns those blanks into NaN
      which we can then impute.

      tenure – fractional months (e.g. 1.0) are confusing; converting to int makes
      the column semantically correct and avoids float precision issues in grouping.

      SeniorCitizen – the value 0 and 1 look numeric, but the column means
      "is this customer a senior?".  If we leave it as int, the scaler will
      treat it as a continuous variable instead of a category, which is wrong.
    """
    df = df.copy()

    # ── TotalCharges: coerce blanks / non-numeric strings to NaN ──────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("Converted TotalCharges from text to numeric (blanks become NaN)")

    # ── tenure: round to nearest int and cast to Int64 (nullable integer) ─────
    # We use Int64 (capital I) so NaN values are preserved during rounding
    df["tenure"] = df["tenure"].round(0).astype("Int64")
    logger.info("Converted tenure from float to integer")

    # ── SeniorCitizen: 0/1 → 'No'/'Yes' ──────────────────────────────────────
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    logger.info("Converted SeniorCitizen from 0/1 to 'No'/'Yes'")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLEAN CATEGORICAL VALUES
# ─────────────────────────────────────────────────────────────────────────────

def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every column that holds strings:
      - Strip leading/trailing whitespace
      - Standardise to Title Case

    WHY: Real datasets often have "Yes ", " yes", "YES" all meaning the same
    thing.  If we leave inconsistent spellings, the encoder treats them as
    different categories, which adds noise and wrong dummy variables.

    Note: "No internet service" and "No phone service" are kept as-is.
    They are meaningful business categories, not errors.
    """
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        # strip whitespace
        df[col] = df[col].str.strip()
        # title-case (preserves "No internet service" → "No Internet Service")
        df[col] = df[col].str.title()
    logger.info(f"Cleaned whitespace/case for {len(str_cols)} string columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. DETECT AND REMOVE DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, id_col: str = "customerID") -> pd.DataFrame:
    """
    Remove:
      1. Fully duplicated rows (every column identical).
      2. Rows with a duplicated customerID after removing full duplicates.

    WHY:
      Duplicate rows add no new information but inflate the training set and
      can bias the model toward patterns seen in repeated rows.

      Duplicated customerIDs after de-duplication indicate data errors (the same
      customer appearing twice with slightly different values).  We keep the
      first occurrence and drop the rest.

    We inspect duplicates before dropping so the decision is transparent.
    """
    df = df.copy()

    # ── Step 1: full-row duplicates ───────────────────────────────────────────
    full_dups = df.duplicated()
    logger.info(f"Found {full_dups.sum()} fully duplicated rows → dropping them")
    df = df[~full_dups]

    # ── Step 2: duplicate customerIDs ─────────────────────────────────────────
    id_dups = df.duplicated(subset=[id_col])
    logger.info(f"Found {id_dups.sum()} rows with a duplicated '{id_col}' → keeping first occurrence")
    df = df.drop_duplicates(subset=[id_col], keep="first")

    logger.info(f"After de-duplication: {len(df)} rows remain")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. HANDLE INVALID NUMERIC VALUES
# ─────────────────────────────────────────────────────────────────────────────

def fix_invalid_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and handle impossible numeric values:
      - Negative tenure or charges → set to NaN (will be imputed later)
      - tenure == 0 AND TotalCharges > 0 → suspicious but possible (first partial month);
        we leave it unless TotalCharges is very large.

    WHY: Negative months or negative charges are physically impossible.
    Leaving them in would distort the model (the scaler would shift its
    mean/std; the tree-based models would create useless split rules).
    """
    df = df.copy()

    # Negative tenure → NaN
    neg_tenure = (df["tenure"] < 0)
    if neg_tenure.sum() > 0:
        logger.info(f"Setting {neg_tenure.sum()} negative tenure values to NaN")
        df.loc[neg_tenure, "tenure"] = np.nan

    # Negative MonthlyCharges → NaN
    neg_monthly = (df["MonthlyCharges"] < 0)
    if neg_monthly.sum() > 0:
        logger.info(f"Setting {neg_monthly.sum()} negative MonthlyCharges values to NaN")
        df.loc[neg_monthly, "MonthlyCharges"] = np.nan

    # Negative TotalCharges → NaN
    neg_total = (df["TotalCharges"] < 0)
    if neg_total.sum() > 0:
        logger.info(f"Setting {neg_total.sum()} negative TotalCharges values to NaN")
        df.loc[neg_total, "TotalCharges"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. DROP UNNECESSARY COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

def drop_unnecessary_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Drop columns that cannot help the model.

    WHY: customerID is a unique identifier assigned by the billing system.
    It has no relationship to churn; including it would only confuse the model
    and bloat the feature space.  Dropping it also removes a privacy concern.
    """
    df = df.copy()
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped columns: {existing}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. FULL CLEANING PIPELINE (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def full_clean(df: pd.DataFrame,
               target_col: str = "Churn",
               id_col: str = "customerID",
               cols_to_drop: list = None) -> pd.DataFrame:
    """
    Run all cleaning steps in order and return a clean DataFrame.

    Steps:
      1. Drop rows with missing target
      2. Fix data types
      3. Clean categorical values (whitespace / case)
      4. Remove duplicates
      5. Fix invalid numeric values
      6. Drop unnecessary columns

    WHY this order matters:
      We drop the missing-target rows first so they don't pollute the
      duplicate check.  We fix dtypes before cleaning categoricals so
      SeniorCitizen is already a string when we apply str.strip().
      We drop unnecessary columns last because they are not harmful
      during the earlier steps.
    """
    if cols_to_drop is None:
        cols_to_drop = [id_col]

    df = drop_missing_target(df, target_col)
    df = fix_data_types(df)
    df = clean_categorical_values(df)
    df = remove_duplicates(df, id_col)
    df = fix_invalid_numeric_values(df)
    df = drop_unnecessary_columns(df, cols_to_drop)

    logger.info(f"Cleaning complete. Final shape: {df.shape}")
    return df
