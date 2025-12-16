import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

# --------------------------------------------------
# Convert data types
# --------------------------------------------------
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types based on their semantics.
    """

    categorical_cols = [
        "hotel",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "deposit_type",
        "customer_type",
        "reservation_status",
    ]

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # Month is temporal but stored as string
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    df["arrival_date_month"] = df["arrival_date_month"].map(month_map)

    # Convert date column
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"],
        errors="coerce"
    )

    return df

# --------------------------------------------------
# Handle invalid values
# --------------------------------------------------
def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle clearly invalid values based on domain knowledge.
    """

    # At least one adult must be present
    df = df[df["adults"] > 0]

    # ADR should not be negative
    df = df[df["adr"] >= 0]

    return df

# --------------------------------------------------
# Handle missing values
# --------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values based on variable semantics and EDA findings.
    """

    # agent & company: not applicable → keep NaN
    # (handled later during feature engineering if needed)

    # country: missing information → assign 'Unknown'
    df["country"] = df["country"].cat.add_categories("Unknown")
    df["country"] = df["country"].fillna("Unknown")

    # children: skewed distribution → impute with median
    children_median = df["children"].median()
    df["children"] = df["children"].fillna(children_median)

    return df

# --------------------------------------------------
# Merge rare categories
# --------------------------------------------------
def merge_rare_categories(
    df: pd.DataFrame,
    categorical_cols: list,
    threshold: float = 0.01,
    protected_cols: list | None = None
) -> pd.DataFrame:
    """
    Merge truly rare categories into 'Other' for selected categorical variables,
    while preserving meaningful categories such as 'Undefined' and 'Unknown'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    categorical_cols : list
        List of categorical columns to consider.
    threshold : float, default=0.01
        Frequency threshold for rare categories.
    protected_cols : list, optional
        Columns for which rare categories should NOT be merged
        due to business meaning.
    """

    df = df.copy()
    n_rows = len(df)

    if protected_cols is None:
        protected_cols = []

    for col in categorical_cols:
        # Ensure column is categorical
        if not isinstance(df[col].dtype, CategoricalDtype):
            continue

        # Skip protected columns (business-critical categories)
        if col in protected_cols:
            continue

        freq = df[col].value_counts(normalize=True)

        # Identify rare categories, excluding semantically meaningful labels
        rare_categories = freq[
            (freq < threshold)
            & (~freq.index.isin(["Undefined", "Unknown"]))
        ].index

        if len(rare_categories) > 0:
            if "Other" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories("Other")

            df[col] = df[col].replace(rare_categories, "Other")

    return df

# --------------------------------------------------
# Validate Dataset 
# --------------------------------------------------
def validate_clean_data(df: pd.DataFrame) -> None:
    """
    Validate cleaned dataset before saving or moving to feature engineering.
    Raises AssertionError if critical issues are found.
    """

    # -----------------------------
    # Shape check
    # -----------------------------
    assert df.shape[0] > 0, "Dataset has no rows."
    assert df.shape[1] > 0, "Dataset has no columns."

    # -----------------------------
    # Critical missing values
    # -----------------------------
    critical_cols = [
        "hotel",
        "is_canceled",
        "lead_time",
        "adr",
        "adults"
    ]

    for col in critical_cols:
        assert df[col].isna().sum() == 0, (
            f"Critical column '{col}' contains missing values."
        )

    # -----------------------------
    # Logical value checks
    # -----------------------------
    assert (df["adults"] > 0).all(), (
        "Found bookings with zero adults."
    )

    assert (df["adr"] >= 0).all(), (
        "Found negative room prices (adr)."
    )

    assert (df["lead_time"] >= 0).all(), (
        "Found negative lead_time values."
    )

    # -----------------------------
    # Data type checks
    # -----------------------------
    assert pd.api.types.is_datetime64_any_dtype(
        df["reservation_status_date"]
    ), "reservation_status_date must be datetime."

    # -----------------------------
    # If all checks pass
    # -----------------------------
    print("Data validation passed. Dataset is clean and consistent.")