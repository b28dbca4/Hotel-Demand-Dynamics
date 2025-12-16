"""
Dashboard for Hotel Demand Analysis
====================================
This module provides dashboard functionality for exploring hotel booking data,
analyzing patterns, and visualizing model results using Plotly.

Features:
- Interactive data exploration with Plotly
- Business metrics dashboard
- Model performance monitoring
- Report generation with interactive HTML outputs
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class HotelDashboard:
    """Main dashboard class for hotel booking analysis using Plotly."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the dashboard.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the data file
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load hotel booking data.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to data file
            
        Returns:
        --------
        data : DataFrame
        """
        if file_path is None:
            file_path = self.data_path
        
        if file_path is None:
            raise ValueError("No data path provided")
        
        self.data = pd.read_csv(file_path)
        
        # Create arrival date
        self.data['arrival_date'] = pd.to_datetime(
            self.data['arrival_date_year'].astype(str) + '-' +
            self.data['arrival_date_month'] + '-' +
            self.data['arrival_date_day_of_month'].astype(str),
            errors='coerce'
        )
        
        return self.data
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the dashboard.
        
        Returns:
        --------
        stats : dict
            Dictionary containing summary statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'total_bookings': len(self.data),
            'total_cancellations': self.data['is_canceled'].sum(),
            'cancellation_rate': self.data['is_canceled'].mean(),
            'avg_adr': self.data['adr'].mean(),
            'avg_lead_time': self.data['lead_time'].mean(),
            'date_range': (self.data['arrival_date'].min(), self.data['arrival_date'].max()),
            'hotel_types': self.data['hotel'].value_counts().to_dict(),
            'market_segments': self.data['market_segment'].value_counts().to_dict(),
        }
        
        return stats
    
    def create_kpi_cards(self, stats: Dict[str, Any]) -> go.Figure:
        """
        Create KPI cards for the dashboard.
        
        Parameters:
        -----------
        stats : dict
            Summary statistics
            
        Returns:
        --------
        fig : plotly Figure
        """
        fig = go.Figure()
        
        # Define KPIs
        kpis = [
            {
                'title': 'Total Bookings',
                'value': f"{stats['total_bookings']:,}",
                'color': '#3498db'
            },
            {
                'title': 'Cancellation Rate',
                'value': f"{stats['cancellation_rate']:.1%}",
                'color': '#e74c3c'
            },
            {
                'title': 'Avg ADR',
                'value': f"€{stats['avg_adr']:.2f}",
                'color': '#2ecc71'
            },
            {
                'title': 'Avg Lead Time',
                'value': f"{stats['avg_lead_time']:.0f} days",
                'color': '#f39c12'
            }
        ]
        
        # Create indicator traces
        for i, kpi in enumerate(kpis):
            fig.add_trace(go.Indicator(
                mode="number",
                value=float(kpi['value'].replace(',', '').replace('€', '').replace('%', '').replace(' days', '')),
                title={'text': kpi['title'], 'font': {'size': 20}},
                number={'font': {'size': 40, 'color': kpi['color']}},
                domain={'x': [i/4, (i+1)/4], 'y': [0, 1]}
            ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            grid={'rows': 1, 'columns': 4, 'pattern': "independent"}
        )
        
        return fig
    
    def create_booking_trends_chart(self) -> go.Figure:
        """
        Create booking trends over time chart.
        
        Returns:
        --------
        fig : plotly Figure
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Group by month
        monthly_data = self.data.groupby(self.data['arrival_date'].dt.to_period('M')).agg({
            'is_canceled': ['count', 'sum', 'mean'],
            'adr': 'mean'
        }).reset_index()
        
        monthly_data.columns = ['month', 'total_bookings', 'cancellations', 'cancel_rate', 'avg_adr']
        monthly_data['month'] = monthly_data['month'].astype(str)
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Bookings Over Time', 'Cancellation Rate & ADR'),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Bookings
        fig.add_trace(
            go.Bar(x=monthly_data['month'], y=monthly_data['total_bookings'],
                  name='Total Bookings', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=monthly_data['month'], y=monthly_data['cancellations'],
                  name='Cancellations', marker_color='coral'),
            row=1, col=1
        )
        
        # Cancellation rate
        fig.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['cancel_rate']*100,
                      name='Cancel Rate %', mode='lines+markers',
                      line=dict(color='red', width=2)),
            row=2, col=1, secondary_y=False
        )
        
        # ADR
        fig.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['avg_adr'],
                      name='Avg ADR (€)', mode='lines+markers',
                      line=dict(color='green', width=2)),
            row=2, col=1, secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Number of Bookings", row=1, col=1)
        fig.update_yaxes(title_text="Cancellation Rate (%)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Average Daily Rate (€)", row=2, col=1, secondary_y=True)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_segment_analysis_chart(self) -> go.Figure:
        """
        Create market segment analysis chart.
        
        Returns:
        --------
        fig : plotly Figure
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        segment_stats = self.data.groupby('market_segment').agg({
            'is_canceled': ['count', 'mean'],
            'adr': 'mean',
            'lead_time': 'mean'
        }).reset_index()
        
        segment_stats.columns = ['segment', 'bookings', 'cancel_rate', 'avg_adr', 'avg_leadtime']
        segment_stats = segment_stats.sort_values('bookings', ascending=True)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Bookings by Segment', 'Cancellation Rate', 'Average ADR'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Bookings
        fig.add_trace(
            go.Bar(y=segment_stats['segment'], x=segment_stats['bookings'],
                  orientation='h', name='Bookings', marker_color='skyblue',
                  text=segment_stats['bookings'], textposition='auto'),
            row=1, col=1
        )
        
        # Cancellation rate
        fig.add_trace(
            go.Bar(y=segment_stats['segment'], x=segment_stats['cancel_rate']*100,
                  orientation='h', name='Cancel Rate %',
                  marker_color='coral',
                  text=[f"{x:.1f}%" for x in segment_stats['cancel_rate']*100],
                  textposition='auto'),
            row=1, col=2
        )
        
        # ADR
        fig.add_trace(
            go.Bar(y=segment_stats['segment'], x=segment_stats['avg_adr'],
                  orientation='h', name='Avg ADR',
                  marker_color='lightgreen',
                  text=[f"€{x:.0f}" for x in segment_stats['avg_adr']],
                  textposition='auto'),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="ADR (€)", row=1, col=3)
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_cancellation_heatmap(self) -> go.Figure:
        """
        Create cancellation heatmap by month and hotel type.
        
        Returns:
        --------
        fig : plotly Figure
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Pivot table
        cancel_pivot = self.data.pivot_table(
            values='is_canceled',
            index='arrival_date_month',
            columns='hotel',
            aggfunc='mean'
        )
        
        # Reorder months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        cancel_pivot = cancel_pivot.reindex(month_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=cancel_pivot.values * 100,
            x=cancel_pivot.columns,
            y=cancel_pivot.index,
            colorscale='RdYlGn_r',
            text=np.round(cancel_pivot.values * 100, 1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 12},
            colorbar=dict(title="Cancel Rate (%)")
        ))
        
        fig.update_layout(
            title='Cancellation Rate Heatmap: Month vs Hotel Type',
            xaxis_title='Hotel Type',
            yaxis_title='Month',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_leadtime_cancellation_chart(self) -> go.Figure:
        """
        Create lead time vs cancellation analysis chart.
        
        Returns:
        --------
        fig : plotly Figure
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Create lead time bins
        bins = [0, 7, 30, 60, 120, 180, 365, 700]
        labels = ['0-7', '8-30', '31-60', '61-120', '121-180', '181-365', '365+']
        
        self.data['leadtime_bin'] = pd.cut(self.data['lead_time'], bins=bins, labels=labels)
        
        leadtime_stats = self.data.groupby(['leadtime_bin', 'hotel']).agg({
            'is_canceled': ['count', 'mean']
        }).reset_index()
        
        leadtime_stats.columns = ['leadtime_bin', 'hotel', 'bookings', 'cancel_rate']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cancellation Rate by Lead Time', 'Booking Volume by Lead Time')
        )
        
        for hotel in self.data['hotel'].unique():
            hotel_data = leadtime_stats[leadtime_stats['hotel'] == hotel]
            
            # Cancellation rate
            fig.add_trace(
                go.Scatter(x=hotel_data['leadtime_bin'], y=hotel_data['cancel_rate']*100,
                          mode='lines+markers', name=f'{hotel}',
                          line=dict(width=3)),
                row=1, col=1
            )
            
            # Booking volume
            fig.add_trace(
                go.Bar(x=hotel_data['leadtime_bin'], y=hotel_data['bookings'],
                      name=f'{hotel}', showlegend=False),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Lead Time (days)", row=1, col=1)
        fig.update_xaxes(title_text="Lead Time (days)", row=1, col=2)
        fig.update_yaxes(title_text="Cancellation Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Bookings", row=1, col=2)
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_deposit_impact_chart(self) -> go.Figure:
        """
        Create deposit type impact on cancellation chart.
        
        Returns:
        --------
        fig : plotly Figure
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        deposit_stats = self.data.groupby('deposit_type').agg({
            'is_canceled': ['count', 'mean']
        }).reset_index()
        
        deposit_stats.columns = ['deposit_type', 'bookings', 'cancel_rate']
        deposit_stats = deposit_stats.sort_values('cancel_rate', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=deposit_stats['deposit_type'],
            y=deposit_stats['cancel_rate'] * 100,
            text=[f"{x:.1f}%" for x in deposit_stats['cancel_rate'] * 100],
            textposition='auto',
            marker_color=['red' if x > 0.5 else 'orange' if x > 0.3 else 'green' 
                         for x in deposit_stats['cancel_rate']],
            name='Cancellation Rate'
        ))
        
        fig.update_layout(
            title='Cancellation Rate by Deposit Type',
            xaxis_title='Deposit Type',
            yaxis_title='Cancellation Rate (%)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_customer_type_analysis(self) -> go.Figure:
        """
        Create customer type analysis chart.
        
        Returns:
        --------
        fig : plotly Figure
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        customer_stats = self.data.groupby('customer_type').agg({
            'is_canceled': ['count', 'mean'],
            'adr': 'mean'
        }).reset_index()
        
        customer_stats.columns = ['customer_type', 'bookings', 'cancel_rate', 'avg_adr']
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=('Booking Distribution', 'Metrics by Customer Type')
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=customer_stats['customer_type'],
                  values=customer_stats['bookings'],
                  hole=0.3),
            row=1, col=1
        )
        
        # Bar chart for cancellation rate
        fig.add_trace(
            go.Bar(x=customer_stats['customer_type'],
                  y=customer_stats['cancel_rate'] * 100,
                  name='Cancel Rate %',
                  marker_color='coral'),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_country_analysis(self, top_n: int = 15) -> go.Figure:
        """
        Create top countries analysis chart.
        
        Parameters:
        -----------
        top_n : int
            Number of top countries to display
            
        Returns:
        --------
        fig : plotly Figure
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        country_stats = self.data.groupby('country').agg({
            'is_canceled': ['count', 'mean']
        }).reset_index()
        
        country_stats.columns = ['country', 'bookings', 'cancel_rate']
        country_stats = country_stats.nlargest(top_n, 'bookings')
        country_stats = country_stats.sort_values('bookings', ascending=True)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Top {top_n} Countries by Bookings', 'Cancellation Rates')
        )
        
        # Bookings
        fig.add_trace(
            go.Bar(y=country_stats['country'], x=country_stats['bookings'],
                  orientation='h', marker_color='lightblue',
                  text=country_stats['bookings'], textposition='auto'),
            row=1, col=1
        )
        
        # Cancellation rate
        fig.add_trace(
            go.Bar(y=country_stats['country'], x=country_stats['cancel_rate']*100,
                  orientation='h', marker_color='salmon',
                  text=[f"{x:.1f}%" for x in country_stats['cancel_rate']*100],
                  textposition='auto'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Bookings", row=1, col=1)
        fig.update_xaxes(title_text="Cancel Rate (%)", row=1, col=2)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_full_dashboard_report(
        self,
        save_path: str = 'hotel_dashboard_report.html',
        include_all: bool = True
    ) -> str:
        """
        Generate a complete dashboard report with all visualizations.
        
        Parameters:
        -----------
        save_path : str
            Path to save the HTML report
        include_all : bool
            Whether to include all charts
            
        Returns:
        --------
        save_path : str
            Path where report was saved
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Get stats
        stats = self.get_summary_stats()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hotel Demand Analytics Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .kpi-container {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 15px;
                    margin-bottom: 30px;
                }}
                .kpi-card {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .kpi-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .kpi-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
                .chart-container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏨 Hotel Demand Analytics Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-value">{stats['total_bookings']:,}</div>
                    <div class="kpi-label">Total Bookings</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{stats['cancellation_rate']:.1%}</div>
                    <div class="kpi-label">Cancellation Rate</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">€{stats['avg_adr']:.2f}</div>
                    <div class="kpi-label">Average ADR</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{stats['avg_lead_time']:.0f}</div>
                    <div class="kpi-label">Avg Lead Time (days)</div>
                </div>
            </div>
        """
        
        if include_all:
            # Add all charts
            charts = []
            
            # Booking trends
            fig = self.create_booking_trends_chart()
            charts.append(("Booking Trends Over Time", fig))
            
            # Segment analysis
            fig = self.create_segment_analysis_chart()
            charts.append(("Market Segment Analysis", fig))
            
            # Cancellation heatmap
            fig = self.create_cancellation_heatmap()
            charts.append(("Cancellation Heatmap", fig))
            
            # Lead time analysis
            fig = self.create_leadtime_cancellation_chart()
            charts.append(("Lead Time vs Cancellation", fig))
            
            # Deposit impact
            fig = self.create_deposit_impact_chart()
            charts.append(("Deposit Impact on Cancellation", fig))
            
            # Customer type
            fig = self.create_customer_type_analysis()
            charts.append(("Customer Type Analysis", fig))
            
            # Country analysis
            fig = self.create_country_analysis(top_n=15)
            charts.append(("Top Countries Analysis", fig))
            
            # Add charts to HTML
            for title, fig in charts:
                chart_html = fig.to_html(include_plotlyjs=False, div_id=title.replace(' ', '_'))
                html_content += f"""
                <div class="chart-container">
                    <h2>{title}</h2>
                    {chart_html}
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save to file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard report saved to: {save_path}")
        return save_path
    
    def show_all_charts(self):
        """
        Display all charts in Jupyter notebook or interactive environment.
        
        Returns:
        --------
        charts : dict
            Dictionary of all generated charts
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        charts = {}
        
        print("Generating dashboard charts...")
        
        # Get stats
        stats = self.get_summary_stats()
        print(f"\n📊 Summary Statistics:")
        print(f"Total Bookings: {stats['total_bookings']:,}")
        print(f"Cancellation Rate: {stats['cancellation_rate']:.1%}")
        print(f"Average ADR: €{stats['avg_adr']:.2f}")
        print(f"Avg Lead Time: {stats['avg_lead_time']:.0f} days\n")
        
        # Generate all charts
        charts['booking_trends'] = self.create_booking_trends_chart()
        charts['segment_analysis'] = self.create_segment_analysis_chart()
        charts['cancellation_heatmap'] = self.create_cancellation_heatmap()
        charts['leadtime_cancellation'] = self.create_leadtime_cancellation_chart()
        charts['deposit_impact'] = self.create_deposit_impact_chart()
        charts['customer_type'] = self.create_customer_type_analysis()
        charts['country_analysis'] = self.create_country_analysis(top_n=15)
        
        # Display charts (works in Jupyter)
        for name, fig in charts.items():
            print(f"\n{'='*50}")
            print(f"Chart: {name.replace('_', ' ').title()}")
            print(f"{'='*50}")
            fig.show()
        
        return charts


if __name__ == "__main__":
    # Example usage
    print("Hotel Dashboard Module")
    print("Usage in Jupyter/Python:"
          "\n  from src.visualization.dashboard import HotelDashboard"
          "\n  dashboard = HotelDashboard('data/raw/hotel_bookings.csv')"
          "\n  dashboard.load_data()"
          "\n  dashboard.show_all_charts()"
          "\n  # OR"
          "\n  dashboard.create_full_dashboard_report('my_report.html')")
