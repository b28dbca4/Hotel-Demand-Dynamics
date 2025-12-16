"""
Visualization Utilities for Hotel Demand Analysis
==================================================
This module provides comprehensive plotting functions for hotel booking data analysis.

Main Categories:
- Distribution plots (histograms, KDE, box plots)
- Time series analysis plots
- Correlation and relationship plots
- Business metrics visualization
- Model performance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set default styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color schemes for different hotel types
HOTEL_COLORS = {
    'City Hotel': '#1f77b4',
    'Resort Hotel': '#ff7f0e'
}

CANCEL_COLORS = {
    0: '#2ecc71',  # Green for not canceled
    1: '#e74c3c'   # Red for canceled
}


class HotelPlotUtils:
    """Utility class for creating hotel booking analysis visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize plotting utilities.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size for matplotlib plots
        style : str
            Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
    @staticmethod
    def setup_plot_style(figsize: Tuple[int, int] = (12, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a figure with consistent styling.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig, ax : matplotlib Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    # ==================== DISTRIBUTION PLOTS ====================
    
    @staticmethod
    def plot_distribution_comparison(
        data: pd.DataFrame,
        feature: str,
        group_col: str,
        bins: int = 50,
        figsize: Tuple[int, int] = (14, 6),
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare distributions of a feature across different groups.
        
        Parameters:
        -----------
        data : DataFrame
            Input data
        feature : str
            Feature to plot
        group_col : str
            Column to group by
        bins : int
            Number of bins for histogram
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        groups = data[group_col].unique()
        
        # Histogram with KDE
        for group in groups:
            subset = data[data[group_col] == group][feature]
            axes[0].hist(subset, bins=bins, alpha=0.6, label=str(group), density=True)
            subset.plot(kind='kde', ax=axes[0], label=f'{group} (KDE)', linewidth=2)
        
        axes[0].set_xlabel(xlabel or feature)
        axes[0].set_ylabel('Density')
        axes[0].set_title(title or f'Distribution of {feature} by {group_col}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data.boxplot(column=feature, by=group_col, ax=axes[1])
        axes[1].set_xlabel(group_col)
        axes[1].set_ylabel(feature)
        axes[1].set_title(f'{feature} Distribution by {group_col}')
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_cancellation_by_leadtime(
        data: pd.DataFrame,
        leadtime_bins: List[int] = [0, 30, 60, 120, 180, 365, 700],
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Analyze cancellation rate by lead time bins.
        
        Parameters:
        -----------
        data : DataFrame
            Hotel booking data with 'lead_time' and 'is_canceled'
        leadtime_bins : list
            Bin edges for lead time categories
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        # Create lead time categories
        data_copy = data.copy()
        data_copy['lead_time_category'] = pd.cut(
            data_copy['lead_time'],
            bins=leadtime_bins,
            labels=[f'{leadtime_bins[i]}-{leadtime_bins[i+1]}' for i in range(len(leadtime_bins)-1)]
        )
        
        # Calculate cancellation rate
        cancel_rate = data_copy.groupby('lead_time_category')['is_canceled'].agg(['mean', 'count'])
        cancel_rate.columns = ['cancellation_rate', 'count']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Cancellation rate by lead time
        axes[0].bar(range(len(cancel_rate)), cancel_rate['cancellation_rate'], 
                   color='coral', alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(len(cancel_rate)))
        axes[0].set_xticklabels(cancel_rate.index, rotation=45, ha='right')
        axes[0].set_ylabel('Cancellation Rate')
        axes[0].set_xlabel('Lead Time (days)')
        axes[0].set_title('Cancellation Rate by Lead Time')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, v in enumerate(cancel_rate['cancellation_rate']):
            axes[0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # Booking count by lead time
        axes[1].bar(range(len(cancel_rate)), cancel_rate['count'], 
                   color='skyblue', alpha=0.7, edgecolor='black')
        axes[1].set_xticks(range(len(cancel_rate)))
        axes[1].set_xticklabels(cancel_rate.index, rotation=45, ha='right')
        axes[1].set_ylabel('Number of Bookings')
        axes[1].set_xlabel('Lead Time (days)')
        axes[1].set_title('Booking Count by Lead Time')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ==================== TIME SERIES PLOTS ====================
    
    @staticmethod
    def plot_time_series_bookings(
        data: pd.DataFrame,
        date_col: str = 'arrival_date',
        freq: str = 'M',
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot booking trends over time.
        
        Parameters:
        -----------
        data : DataFrame
            Booking data with date column
        date_col : str
            Name of date column
        freq : str
            Frequency for resampling ('D', 'W', 'M', etc.)
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        data_copy = data.copy()
        
        # Ensure date column is datetime
        if date_col in data_copy.columns:
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
        else:
            # Create date from year, month, day columns
            data_copy['arrival_date'] = pd.to_datetime(
                data_copy['arrival_date_year'].astype(str) + '-' +
                data_copy['arrival_date_month'] + '-' +
                data_copy['arrival_date_day_of_month'].astype(str),
                errors='coerce'
            )
        
        data_copy = data_copy.set_index('arrival_date').sort_index()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Total bookings over time
        bookings_ts = data_copy.resample(freq).size()
        axes[0].plot(bookings_ts.index, bookings_ts.values, linewidth=2, marker='o', markersize=4)
        axes[0].set_ylabel('Number of Bookings')
        axes[0].set_title(f'Total Bookings Over Time ({freq} frequency)')
        axes[0].grid(True, alpha=0.3)
        
        # Cancellation rate over time
        cancel_ts = data_copy.resample(freq)['is_canceled'].mean()
        axes[1].plot(cancel_ts.index, cancel_ts.values, linewidth=2, marker='o', 
                    markersize=4, color='red')
        axes[1].set_ylabel('Cancellation Rate')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Cancellation Rate Over Time')
        axes[1].grid(True, alpha=0.3)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_seasonal_patterns(
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot seasonal booking patterns.
        
        Parameters:
        -----------
        data : DataFrame
            Hotel booking data
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Monthly patterns
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        monthly_bookings = data.groupby('arrival_date_month').size().reindex(month_order)
        axes[0, 0].bar(range(12), monthly_bookings.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_xticks(range(12))
        axes[0, 0].set_xticklabels([m[:3] for m in month_order], rotation=45)
        axes[0, 0].set_ylabel('Number of Bookings')
        axes[0, 0].set_title('Bookings by Month')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Monthly cancellation rate
        monthly_cancel = data.groupby('arrival_date_month')['is_canceled'].mean().reindex(month_order)
        axes[0, 1].plot(range(12), monthly_cancel.values, marker='o', linewidth=2, color='coral')
        axes[0, 1].set_xticks(range(12))
        axes[0, 1].set_xticklabels([m[:3] for m in month_order], rotation=45)
        axes[0, 1].set_ylabel('Cancellation Rate')
        axes[0, 1].set_title('Cancellation Rate by Month')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Weekly patterns (week of year)
        weekly_bookings = data.groupby('arrival_date_week_number').size()
        axes[1, 0].plot(weekly_bookings.index, weekly_bookings.values, linewidth=2)
        axes[1, 0].set_xlabel('Week of Year')
        axes[1, 0].set_ylabel('Number of Bookings')
        axes[1, 0].set_title('Bookings by Week of Year')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ADR by month
        monthly_adr = data.groupby('arrival_date_month')['adr'].mean().reindex(month_order)
        axes[1, 1].bar(range(12), monthly_adr.values, color='green', alpha=0.6)
        axes[1, 1].set_xticks(range(12))
        axes[1, 1].set_xticklabels([m[:3] for m in month_order], rotation=45)
        axes[1, 1].set_ylabel('Average Daily Rate (ADR)')
        axes[1, 1].set_title('ADR by Month')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ==================== CORRELATION PLOTS ====================
    
    @staticmethod
    def plot_correlation_heatmap(
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        annot: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation heatmap for numerical features.
        
        Parameters:
        -----------
        data : DataFrame
            Input data
        features : list, optional
            List of features to include. If None, use all numeric columns
        figsize : tuple
            Figure size
        annot : bool
            Whether to annotate cells with correlation values
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = data[features].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ==================== BUSINESS METRICS PLOTS ====================
    
    @staticmethod
    def plot_segment_analysis(
        data: pd.DataFrame,
        segment_col: str = 'market_segment',
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Analyze bookings by market segment.
        
        Parameters:
        -----------
        data : DataFrame
            Hotel booking data
        segment_col : str
            Column name for market segment
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Booking count by segment
        segment_counts = data[segment_col].value_counts()
        axes[0, 0].barh(range(len(segment_counts)), segment_counts.values, color='skyblue')
        axes[0, 0].set_yticks(range(len(segment_counts)))
        axes[0, 0].set_yticklabels(segment_counts.index)
        axes[0, 0].set_xlabel('Number of Bookings')
        axes[0, 0].set_title('Bookings by Market Segment')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Cancellation rate by segment
        segment_cancel = data.groupby(segment_col)['is_canceled'].mean().sort_values(ascending=False)
        colors = ['red' if x > 0.4 else 'orange' if x > 0.25 else 'green' 
                 for x in segment_cancel.values]
        axes[0, 1].barh(range(len(segment_cancel)), segment_cancel.values, color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(len(segment_cancel)))
        axes[0, 1].set_yticklabels(segment_cancel.index)
        axes[0, 1].set_xlabel('Cancellation Rate')
        axes[0, 1].set_title('Cancellation Rate by Market Segment')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # ADR by segment
        segment_adr = data.groupby(segment_col)['adr'].mean().sort_values(ascending=False)
        axes[1, 0].barh(range(len(segment_adr)), segment_adr.values, color='green', alpha=0.6)
        axes[1, 0].set_yticks(range(len(segment_adr)))
        axes[1, 0].set_yticklabels(segment_adr.index)
        axes[1, 0].set_xlabel('Average Daily Rate (€)')
        axes[1, 0].set_title('ADR by Market Segment')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Lead time by segment
        segment_leadtime = data.groupby(segment_col)['lead_time'].mean().sort_values(ascending=False)
        axes[1, 1].barh(range(len(segment_leadtime)), segment_leadtime.values, 
                       color='purple', alpha=0.6)
        axes[1, 1].set_yticks(range(len(segment_leadtime)))
        axes[1, 1].set_yticklabels(segment_leadtime.index)
        axes[1, 1].set_xlabel('Average Lead Time (days)')
        axes[1, 1].set_title('Lead Time by Market Segment')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_hotel_comparison(
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare City Hotel vs Resort Hotel metrics.
        
        Parameters:
        -----------
        data : DataFrame
            Hotel booking data
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        hotels = data['hotel'].unique()
        colors = [HOTEL_COLORS.get(h, 'gray') for h in hotels]
        
        # Booking counts
        hotel_counts = data['hotel'].value_counts()
        axes[0, 0].bar(range(len(hotel_counts)), hotel_counts.values, color=colors, alpha=0.7)
        axes[0, 0].set_xticks(range(len(hotel_counts)))
        axes[0, 0].set_xticklabels(hotel_counts.index, rotation=15)
        axes[0, 0].set_ylabel('Number of Bookings')
        axes[0, 0].set_title('Total Bookings by Hotel Type')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Cancellation rate
        hotel_cancel = data.groupby('hotel')['is_canceled'].mean()
        axes[0, 1].bar(range(len(hotel_cancel)), hotel_cancel.values, color=colors, alpha=0.7)
        axes[0, 1].set_xticks(range(len(hotel_cancel)))
        axes[0, 1].set_xticklabels(hotel_cancel.index, rotation=15)
        axes[0, 1].set_ylabel('Cancellation Rate')
        axes[0, 1].set_title('Cancellation Rate by Hotel Type')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # ADR distribution
        for hotel in hotels:
            hotel_data = data[data['hotel'] == hotel]['adr']
            hotel_data[hotel_data <= 500].plot(kind='kde', ax=axes[1, 0], 
                                               label=hotel, linewidth=2,
                                               color=HOTEL_COLORS.get(hotel, 'gray'))
        axes[1, 0].set_xlabel('Average Daily Rate (€)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('ADR Distribution by Hotel Type')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Lead time distribution
        for hotel in hotels:
            hotel_data = data[data['hotel'] == hotel]['lead_time']
            hotel_data[hotel_data <= 365].plot(kind='kde', ax=axes[1, 1], 
                                               label=hotel, linewidth=2,
                                               color=HOTEL_COLORS.get(hotel, 'gray'))
        axes[1, 1].set_xlabel('Lead Time (days)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Lead Time Distribution by Hotel Type')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ==================== MODEL EVALUATION PLOTS ====================
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix for classification results.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list, optional
            Class labels
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = ['Not Canceled', 'Canceled']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance from a trained model.
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        importances : array
            Feature importance values
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_importances, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve for binary classification.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred_proba : array-like
            Predicted probabilities for positive class
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib Figure
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# ==================== PLOTLY INTERACTIVE PLOTS ====================

class InteractivePlots:
    """Class for creating interactive Plotly visualizations."""
    
    @staticmethod
    def create_cancellation_funnel(
        data: pd.DataFrame,
        stages: List[str] = ['Total Bookings', 'Not Canceled', 'Canceled'],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a funnel chart for cancellation analysis.
        
        Parameters:
        -----------
        data : DataFrame
            Hotel booking data
        stages : list
            Stages for funnel
        save_path : str, optional
            Path to save HTML figure
            
        Returns:
        --------
        fig : plotly Figure
        """
        total = len(data)
        not_canceled = len(data[data['is_canceled'] == 0])
        canceled = len(data[data['is_canceled'] == 1])
        
        fig = go.Figure(go.Funnel(
            y=stages,
            x=[total, not_canceled, canceled],
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(
            title='Booking Cancellation Funnel',
            font=dict(size=14)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def create_interactive_time_series(
        data: pd.DataFrame,
        date_col: str = 'arrival_date',
        value_cols: List[str] = ['bookings', 'cancellations'],
        title: str = 'Hotel Bookings Time Series',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive time series plot.
        
        Parameters:
        -----------
        data : DataFrame
            Time series data
        date_col : str
            Date column name
        value_cols : list
            Columns to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save HTML figure
            
        Returns:
        --------
        fig : plotly Figure
        """
        fig = go.Figure()
        
        for col in value_cols:
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data[date_col],
                    y=data[col],
                    mode='lines+markers',
                    name=col.replace('_', ' ').title()
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Count',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def create_scatter_matrix(
        data: pd.DataFrame,
        features: List[str],
        color_by: Optional[str] = None,
        title: str = 'Feature Scatter Matrix',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive scatter plot matrix.
        
        Parameters:
        -----------
        data : DataFrame
            Input data
        features : list
            Features to include
        color_by : str, optional
            Column to color points by
        title : str
            Plot title
        save_path : str, optional
            Path to save HTML figure
            
        Returns:
        --------
        fig : plotly Figure
        """
        fig = px.scatter_matrix(
            data,
            dimensions=features,
            color=color_by,
            title=title,
            height=800
        )
        
        fig.update_traces(diagonal_visible=False)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
