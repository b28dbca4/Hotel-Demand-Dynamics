import pandas as pd

# =====================================================
# IQR-based outlier detection
# =====================================================
def count_outliers_iqr(series):
    """
    Count outliers in a pandas Series using the IQR method.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return ((series < lower) | (series > upper)).sum()

def summarize_outliers_iqr(df, cols):
    """
    Count IQR-based outliers for multiple columns
    and return a summary DataFrame.
    """
    outlier_counts = {
        col: count_outliers_iqr(df[col].dropna())
        for col in cols
    }

    outlier_df = (
        pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["outlier_count"])
        .rename_axis("variable")
        .reset_index()
    )

    outlier_df["outlier_ratio (%)"] = (
        outlier_df["outlier_count"] / len(df) * 100
    ).round(2)

    return outlier_df

# =====================================================
# Missing values
# =====================================================
def get_missing_percentage(df, cols):
    """
    Calculate missing value percentage for selected columns.
    """
    missing_df = (
        df[cols]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )
    missing_df.columns = ["variable", "missing_percentage (%)"]
    return missing_df

# =====================================================
# Impossible / invalid numeric values
# =====================================================
def check_invalid_numeric_values(df):
    """
    Check for invalid values in numerical columns
    based on practical data meaning.
    """
    invalid_checks = {
        # Variables that should not be negative
        "negative_counts": (
            df[[
                "lead_time",
                "stays_in_week_nights",
                "stays_in_weekend_nights",
                "children",
                "babies",
                "previous_cancellations",
                "previous_bookings_not_canceled",
                "booking_changes",
                "days_in_waiting_list",
                "required_car_parking_spaces",
                "total_of_special_requests"
            ]] < 0
        ).sum().sum(),

        # Adults must be strictly greater than 0
        "invalid_adults (<=0)": (df["adults"] <= 0).sum(),

        # Binary variables must be 0 or 1
        "invalid_binary": (
            ~df[["is_canceled", "is_repeated_guest"]].isin([0, 1])
        ).sum().sum(),

        # Price should not be negative
        "negative_adr": (df["adr"] < 0).sum()
    }

    return invalid_checks

# =====================================================
# Categorical data utilities
# =====================================================
def check_category_inconsistencies(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Check inconsistencies in categorical columns by comparing
    original values with normalized values.

    Normalization handles:
    - lowercase
    - leading/trailing spaces
    - multiple spaces
    - separators such as '-', '/', '_'
    """

    records = []

    for col in cols:
        original = df[col].dropna().astype(str)

        normalized = (
            original
            .str.lower()
            .str.strip()
            .str.replace(r"[\s\-/]+", "_", regex=True)
        )

        if original.nunique() != normalized.nunique():
            records.append({
                "variable": col,
                "original_unique": original.nunique(),
                "normalized_unique": normalized.nunique(),
                "potential_inconsistency": original.nunique() - normalized.nunique()
            })

    return pd.DataFrame(records)

def get_unexpected_categories(df, cols):
    """
    Identify unexpected category values discovered during EDA.
    """
    records = []

    for col in cols:
        count = (df[col] == "Undefined").sum()
        if count > 0:
            records.append({
                "variable": col,
                "unexpected_value": "Undefined",
                "count": count,
                "percentage (%)": round(count / len(df) * 100, 2)
            })

    return pd.DataFrame(records)

def get_rare_categories(
    df: pd.DataFrame,
    cols: list,
    threshold: float = 0.01,
    max_per_variable: int = 10
) -> pd.DataFrame:
    total = len(df)
    rare_info = []

    for col in cols:
        value_counts = df[col].value_counts(dropna=False)
        rare = value_counts[value_counts / total < threshold]

        for category, count in rare.items():
            rare_info.append({
                "variable": col,
                "category": category,
                "count": count,
                "percentage (%)": round(count / total * 100, 2)
            })

    df_rare = (
        pd.DataFrame(rare_info)
        .sort_values(["variable", "percentage (%)"])
        .groupby("variable")
        .head(max_per_variable)
        .reset_index(drop=True)
    )

    return df_rare

# =====================================================
# Missing Summary 
# =====================================================
def get_missing_summary(df, cols):
    """
    Missing summary for selected columns: count + percentage.
    """
    summary = (
        df[cols]
        .isna()
        .agg(["sum", "mean"])
        .T
        .reset_index()
    )
    summary.columns = ["variable", "missing_count", "missing_ratio"]
    summary["missing_percentage (%)"] = (summary["missing_ratio"] * 100).round(2)
    summary = summary.drop(columns=["missing_ratio"]).sort_values(
        "missing_percentage (%)", ascending=False
    ).reset_index(drop=True)

    return summary
