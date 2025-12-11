
import pandas as pd
import numpy as np
from typing import Optional

# -------------------------
# Helper: Excel serial -> datetime
# -------------------------
def excel_serial_to_datetime(serial):
    """Convert Excel serial date (float/int) to Timestamp; returns pd.NaT for non-numeric."""
    try:
        serial = float(serial)
    except Exception:
        return pd.NaT
    if serial <= 0:
        return pd.NaT
    # Excel's serial number origin aligned with '1899-12-30' for compatibility
    return pd.Timestamp('1899-12-30') + pd.to_timedelta(serial, unit='D')

# -------------------------
# Helper: robust parsing for a pandas Series
# -------------------------
def robust_to_datetime(series: pd.Series, dayfirst: Optional[bool]=None) -> pd.Series:
    """
    Parse a series of potentially mixed-format date values into datetimes.
    Tries inferring formats, toggles dayfirst if many values are NaT,
    and handles numeric Excel serials as a fallback.
    """
    out = pd.to_datetime(series, errors='coerce', infer_datetime_format=True, dayfirst=dayfirst)

    # If a large fraction failed and dayfirst wasn't specified, try the alternate
    if out.isna().mean() > 0.15 and dayfirst is None:
        alt = pd.to_datetime(series, errors='coerce', infer_datetime_format=True, dayfirst=not True)
        out = out.where(out.notna(), alt)

    # Handle numeric Excel serials
    mask_num = out.isna() & series.apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))
    if mask_num.any():
        out_num = series[mask_num].map(excel_serial_to_datetime)
        out.loc[mask_num] = out_num

    # Try to remove tz info if present
    try:
        out = out.dt.tz_convert(None)
    except Exception:
        pass

    return out

# -------------------------
# Main preprocessing function
# -------------------------
def preprocess_visa_csv(
    path: str,
    submission_col: str = 'submission_date',
    decision_col: str = 'decision_date',
    id_col: Optional[str] = 'application_id',
    drop_rows_with_missing_dates: bool = True,
    drop_negative_processing: bool = True,
    categorical_fill: str = 'Unknown',
    numeric_fill_strategy: str = 'median'  # options: 'median', 'mean', 'zero'
) -> pd.DataFrame:
    """
    Load CSV, clean data, parse dates robustly, compute processing_days (target),
    perform simple imputation, and return cleaned DataFrame.
    """
    # Load
    df = pd.read_csv(path)

    # Normalize column names to a safe snake_case style
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    submission_col = submission_col.strip().lower().replace(' ', '_')
    decision_col = decision_col.strip().lower().replace(' ', '_')
    if id_col:
        id_col = id_col.strip().lower().replace(' ', '_')

    # Trim whitespace and normalize text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': None})

    # Ensure required columns exist
    if submission_col not in df.columns or decision_col not in df.columns:
        raise KeyError(f"Missing required date columns. Available columns: {list(df.columns)}")

    # Robust parse
    df[submission_col + '_parsed'] = robust_to_datetime(df[submission_col])
    df[decision_col + '_parsed']   = robust_to_datetime(df[decision_col])

    # Flag missing dates
    missing_dates_mask = df[submission_col + '_parsed'].isna() | df[decision_col + '_parsed'].isna()
    if missing_dates_mask.any():
        print(f"Warning: {missing_dates_mask.sum()} rows have missing or malformed dates.")
        df['_missing_dates'] = missing_dates_mask
        if drop_rows_with_missing_dates:
            df = df.loc[~missing_dates_mask].copy()

    # Compute target: processing_days (decision - submission) in integer days
    df['processing_days'] = (df[decision_col + '_parsed'] - df[submission_col + '_parsed']).dt.days

    # Handle negative and zero processing days
    neg_mask = df['processing_days'] < 0
    zero_mask = df['processing_days'] == 0
    if neg_mask.any():
        print(f"Warning: {neg_mask.sum()} rows have negative processing days (decision before submission).")
        df['_negative_processing'] = neg_mask
        if drop_negative_processing:
            df = df.loc[~neg_mask].copy()
    if zero_mask.any():
        print(f"Note: {zero_mask.sum()} rows have zero processing days (same-day decisions).")

    # Simple imputation for categorical columns (except parsed date columns)
    for col in df.select_dtypes(include=['object']).columns:
        if col.endswith('_parsed'):
            continue
        df[col] = df[col].fillna(categorical_fill)

    # Simple imputation for numeric columns except processing_days
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['processing_days']]
    for col in numeric_cols:
        if df[col].isna().any():
            if numeric_fill_strategy == 'median':
                fill_val = df[col].median(skipna=True)
            elif numeric_fill_strategy == 'mean':
                fill_val = df[col].mean(skipna=True)
            elif numeric_fill_strategy == 'zero':
                fill_val = 0
            else:
                fill_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(fill_val)

    # Reorder columns for readability: id, original dates, parsed dates, target, rest, flags
    keep_cols = []
    if id_col and id_col in df.columns:
        keep_cols.append(id_col)
    keep_cols += [submission_col, submission_col + '_parsed',
                  decision_col, decision_col + '_parsed',
                  'processing_days']
    extras = [c for c in df.columns if c not in keep_cols and not c.startswith('_')]
    keep_cols += extras
    flags = [c for c in df.columns if c.startswith('_')]
    keep_cols += flags

    cleaned = df[keep_cols].reset_index(drop=True)
    return cleaned

# -------------------------
# Script entrypoint
# -------------------------
if __name__ == '__main__':
    INPUT_CSV = 'visa_data_100.csv'           # change path if needed
    OUTPUT_CSV = 'visa_data_100_cleaned.csv'  # final output (contains processing_days)

    cleaned_df = preprocess_visa_csv(
        INPUT_CSV,
        submission_col='submission_date',
        decision_col='decision_date',
        id_col='application_id',
        drop_rows_with_missing_dates=True,
        drop_negative_processing=True,
        categorical_fill='Unknown',
        numeric_fill_strategy='median'
    )

    # Quick report (useful to inspect what's happened)
    print("=== Preprocessing report ===")
    try:
        input_rows = len(pd.read_csv(INPUT_CSV))
    except Exception:
        input_rows = "unknown"
    print(f"Input rows: {input_rows}")
    print(f"Output rows (after drops): {len(cleaned_df)}")
    print("\nProcessing days summary:")
    print(cleaned_df['processing_days'].describe())

    # Show first 5 rows with the computed target
    print("\nSample rows (submission_date, decision_date, processing_days):")
    display_cols = [c for c in cleaned_df.columns if c in ['submission_date_parsed', 'decision_date_parsed', 'processing_days']]
    if display_cols:
        print(cleaned_df[display_cols].head())

    # Save cleaned CSV
    cleaned_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCleaned dataset saved to: {OUTPUT_CSV}")
