"""
Visualization Package for Hotel Demand Analysis
================================================

This package provides comprehensive visualization tools for hotel booking data analysis,
including static plots, interactive dashboards, and business intelligence visualizations.

Modules:
--------
- plot_utils: Static plotting utilities using matplotlib and seaborn
- dashboard: Interactive dashboard using Plotly and Streamlit

Classes:
--------
- HotelPlotUtils: Static plotting utility class
- InteractivePlots: Interactive Plotly visualizations
- HotelDashboard: Main dashboard class

Usage Examples:
---------------

1. Using static plots:
```python
from visualization import HotelPlotUtils
import pandas as pd

# Load data
data = pd.read_csv('hotel_bookings.csv')

# Initialize plotting utilities
plotter = HotelPlotUtils(figsize=(12, 6))

# Create distribution comparison
fig = plotter.plot_distribution_comparison(
    data, 
    feature='lead_time',
    group_col='is_canceled'
)

# Create cancellation analysis
fig = plotter.plot_cancellation_by_leadtime(data)

# Create time series
fig = plotter.plot_time_series_bookings(data)

# Create correlation heatmap
fig = plotter.plot_correlation_heatmap(data)
```

2. Using interactive plots:
```python
from visualization import InteractivePlots

interactive = InteractivePlots()

# Create interactive time series
fig = interactive.create_interactive_time_series(
    data,
    date_col='arrival_date',
    value_cols=['bookings', 'cancellations']
)
fig.show()

# Create scatter matrix
fig = interactive.create_scatter_matrix(
    data,
    features=['lead_time', 'adr', 'stays_in_weekend_nights'],
    color_by='is_canceled'
)
```

3. Running the dashboard:
```python
from visualization import HotelDashboard

# Initialize dashboard
dashboard = HotelDashboard(data_path='data/raw/hotel_bookings.csv')

# Load data
dashboard.load_data()

# Get summary statistics
stats = dashboard.get_summary_stats()

# Create various charts
kpi_fig = dashboard.create_kpi_cards(stats)
trend_fig = dashboard.create_booking_trends_chart()
segment_fig = dashboard.create_segment_analysis_chart()
```

4. Generate HTML dashboard report:
```python
from visualization import HotelDashboard

dashboard = HotelDashboard(data_path='data/raw/hotel_bookings.csv')
dashboard.load_data()

# Show all charts in Jupyter
dashboard.show_all_charts()

# Or generate HTML report
dashboard.create_full_dashboard_report('reports/dashboard_report.html')
```

Constants:
----------
- HOTEL_COLORS: Color mapping for hotel types
- CANCEL_COLORS: Color mapping for cancellation status

Author: Hotel Demand Analytics Team
Version: 1.0.0
"""

from .plot_utils import (
    HotelPlotUtils,
    InteractivePlots,
    HOTEL_COLORS,
    CANCEL_COLORS
)

from .dashboard import HotelDashboard

__all__ = [
    # Classes
    'HotelPlotUtils',
    'InteractivePlots',
    'HotelDashboard',
    
    # Constants
    'HOTEL_COLORS',
    'CANCEL_COLORS',
]

__version__ = '1.0.0'
__author__ = 'Hotel Demand Analytics Team'

# Module level convenience functions

def quick_plot_overview(data, save_dir=None):
    """
    Quick function to generate all overview plots.
    
    Parameters:
    -----------
    data : DataFrame
        Hotel booking data
    save_dir : str, optional
        Directory to save plots
        
    Returns:
    --------
    figures : dict
        Dictionary of matplotlib/plotly figures
    """
    plotter = HotelPlotUtils()
    figures = {}
    
    # Distribution plots
    figures['leadtime_dist'] = plotter.plot_distribution_comparison(
        data, 'lead_time', 'is_canceled',
        save_path=f"{save_dir}/leadtime_distribution.png" if save_dir else None
    )
    
    figures['adr_dist'] = plotter.plot_distribution_comparison(
        data, 'adr', 'hotel',
        save_path=f"{save_dir}/adr_distribution.png" if save_dir else None
    )
    
    # Time series
    figures['time_series'] = plotter.plot_time_series_bookings(
        data,
        save_path=f"{save_dir}/time_series.png" if save_dir else None
    )
    
    # Seasonal patterns
    figures['seasonal'] = plotter.plot_seasonal_patterns(
        data,
        save_path=f"{save_dir}/seasonal_patterns.png" if save_dir else None
    )
    
    # Business metrics
    figures['segment'] = plotter.plot_segment_analysis(
        data,
        save_path=f"{save_dir}/segment_analysis.png" if save_dir else None
    )
    
    figures['hotel_comparison'] = plotter.plot_hotel_comparison(
        data,
        save_path=f"{save_dir}/hotel_comparison.png" if save_dir else None
    )
    
    return figures


def quick_model_evaluation(y_true, y_pred, y_pred_proba=None, 
                           feature_names=None, feature_importances=None,
                           save_dir=None):
    """
    Quick function to generate all model evaluation plots.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities
    feature_names : list, optional
        Feature names
    feature_importances : array-like, optional
        Feature importance values
    save_dir : str, optional
        Directory to save plots
        
    Returns:
    --------
    figures : dict
        Dictionary of matplotlib figures
    """
    plotter = HotelPlotUtils()
    figures = {}
    
    # Confusion matrix
    figures['confusion_matrix'] = plotter.plot_confusion_matrix(
        y_true, y_pred,
        save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None
    )
    
    # ROC curve
    if y_pred_proba is not None:
        figures['roc_curve'] = plotter.plot_roc_curve(
            y_true, y_pred_proba,
            save_path=f"{save_dir}/roc_curve.png" if save_dir else None
        )
    
    # Feature importance
    if feature_names is not None and feature_importances is not None:
        figures['feature_importance'] = plotter.plot_feature_importance(
            feature_names, feature_importances,
            save_path=f"{save_dir}/feature_importance.png" if save_dir else None
        )
    
    return figures


def create_interactive_dashboard(data_path=None):
    """
    Quick function to create and return a dashboard instance.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to hotel booking data
        
    Returns:
    --------
    dashboard : HotelDashboard
        Dashboard instance
    """
    dashboard = HotelDashboard(data_path=data_path)
    
    if data_path:
        dashboard.load_data()
    
    return dashboard


# Print package info when imported
def _print_info():
    """Print package information."""
    print(f"Hotel Visualization Package v{__version__}")
    print(f"Available classes: {', '.join([c for c in __all__ if c[0].isupper()])}")


# Uncomment to show info on import
# _print_info()
