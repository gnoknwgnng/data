#!/usr/bin/env python3
"""
Interactive Dashboard Generator
Creates comprehensive HTML dashboards with multiple visualization types
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DashboardGenerator:
    """
    Generate interactive HTML dashboards
    """
    
    def __init__(self):
        self.dashboards = {}
        self.config = {
            'theme': 'plotly_white',
            'height': 800,
            'width': 1200
        }
    
    def create_summary_dashboard(self, data: pd.DataFrame, 
                               analysis_results: Dict[str, Any],
                               save_path: str = None) -> str:
        """
        Create a comprehensive summary dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Data Overview', 'Missing Data',
                'Numeric Distributions', 'Categorical Analysis',
                'Correlation Matrix', 'Quality Metrics'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Data Overview (Indicator)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=len(data),
                title={'text': "Total Rows"},
                delta={'reference': len(data) * 0.9},
                gauge={'axis': {'range': [None, len(data)]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, len(data) * 0.6], 'color': "lightgray"},
                                {'range': [len(data) * 0.6, len(data) * 0.8], 'color': "gray"},
                                {'range': [len(data) * 0.8, len(data)], 'color': "darkgray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': len(data) * 0.9}}
            ),
            row=1, col=1
        )
        
        # 2. Missing Data (Bar chart)
        missing_data = data.isnull().sum()
        fig.add_trace(
            go.Bar(
                x=list(missing_data.index),
                y=list(missing_data.values),
                name="Missing Values",
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # 3. Numeric Distributions (Histogram)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            for col in numeric_columns[:3]:  # Show first 3 numeric columns
                fig.add_trace(
                    go.Histogram(
                        x=data[col].dropna(),
                        name=col,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
        
        # 4. Categorical Analysis (Bar chart)
        categorical_columns = data.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns[:3]:  # Show first 3 categorical columns
                value_counts = data[col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(
                        x=list(value_counts.index),
                        y=list(value_counts.values),
                        name=col
                    ),
                    row=2, col=2
                )
        
        # 5. Correlation Matrix (Heatmap)
        if len(numeric_columns) > 1:
            correlation_matrix = data[numeric_columns].corr()
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=3, col=1
            )
        
        # 6. Quality Score (Indicator)
        quality_score = analysis_results.get('summary', {}).get('quality_score', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_score,
                title={'text': "Data Quality Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "red"},
                                {'range': [50, 70], 'color': "yellow"},
                                {'range': [70, 90], 'color': "lightgreen"},
                                {'range': [90, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Comprehensive Data Analysis Dashboard",
            height=1000,
            showlegend=True,
            template=self.config['theme']
        )
        
        # Save to HTML
        if save_path:
            fig.write_html(f"{save_path}/summary_dashboard.html")
        
        return fig
    
    def create_advanced_dashboard(self, data: pd.DataFrame, 
                                advanced_results: Dict[str, Any],
                                save_path: str = None) -> str:
        """
        Create advanced analytics dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Anomaly Detection', 'Clustering Results',
                'PCA Analysis', 'Temporal Trends'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Anomaly Detection
        if 'anomaly_detection' in advanced_results and 'error' not in advanced_results['anomaly_detection']:
            anomaly_data = advanced_results['anomaly_detection']
            if 'anomaly_scores' in anomaly_data:
                # Create scatter plot of anomaly scores
                indices = range(len(anomaly_data['anomaly_scores']))
                fig.add_trace(
                    go.Scatter(
                        x=indices,
                        y=anomaly_data['anomaly_scores'],
                        mode='markers',
                        name='Anomaly Scores',
                        marker=dict(
                            size=8,
                            color=anomaly_data['anomaly_scores'],
                            colorscale='Reds',
                            showscale=True
                        )
                    ),
                    row=1, col=1
                )
        
        # 2. Clustering Results
        if 'clustering' in advanced_results and 'error' not in advanced_results['clustering']:
            cluster_data = advanced_results['clustering']
            if 'cluster_labels' in cluster_data and 'pca_analysis' in advanced_results:
                pca_data = advanced_results['pca_analysis']
                if 'pca_result' in pca_data:
                    pca_result = np.array(pca_data['pca_result'])
                    cluster_labels = cluster_data['cluster_labels']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pca_result[:, 0],
                            y=pca_result[:, 1],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=cluster_labels,
                                colorscale='Viridis',
                                showscale=True
                            ),
                            name='Clusters'
                        ),
                        row=1, col=2
                    )
        
        # 3. PCA Analysis
        if 'pca_analysis' in advanced_results and 'error' not in advanced_results['pca_analysis']:
            pca_data = advanced_results['pca_analysis']
            if 'explained_variance_ratio' in pca_data:
                fig.add_trace(
                    go.Bar(
                        x=[f'PC{i+1}' for i in range(len(pca_data['explained_variance_ratio']))],
                        y=pca_data['explained_variance_ratio'],
                        name='Explained Variance'
                    ),
                    row=2, col=1
                )
        
        # 4. Temporal Trends
        if 'temporal_analysis' in advanced_results and 'error' not in advanced_results['temporal_analysis']:
            temporal_data = advanced_results['temporal_analysis']
            if 'data_points' in temporal_data:
                dates = [point['date'] for point in temporal_data['data_points']]
                values = [point['value'] for point in temporal_data['data_points']]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name='Temporal Trend'
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Advanced Analytics Dashboard",
            height=800,
            showlegend=True,
            template=self.config['theme']
        )
        
        # Save to HTML
        if save_path:
            fig.write_html(f"{save_path}/advanced_dashboard.html")
        
        return fig
    
    def create_quality_dashboard(self, data: pd.DataFrame, 
                               quality_results: Dict[str, Any],
                               save_path: str = None) -> str:
        """
        Create data quality focused dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Data Quality Score', 'Missing Data Heatmap',
                'Outlier Analysis', 'Data Type Distribution'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "heatmap"}],
                [{"type": "box"}, {"type": "pie"}]
            ]
        )
        
        # 1. Quality Score
        quality_score = quality_results.get('quality_score', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_score,
                title={'text': "Overall Quality Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "red"},
                                {'range': [50, 70], 'color': "orange"},
                                {'range': [70, 90], 'color': "yellow"},
                                {'range': [90, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=1, col=1
        )
        
        # 2. Missing Data Heatmap
        missing_matrix = data.isnull().astype(int)
        fig.add_trace(
            go.Heatmap(
                z=missing_matrix.values,
                x=data.columns,
                y=range(len(data)),
                colorscale='Reds',
                name='Missing Data'
            ),
            row=1, col=2
        )
        
        # 3. Outlier Analysis (Box plots)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            for col in numeric_columns[:3]:  # Show first 3 numeric columns
                fig.add_trace(
                    go.Box(
                        y=data[col].dropna(),
                        name=col
                    ),
                    row=2, col=1
                )
        
        # 4. Data Type Distribution
        data_types = data.dtypes.value_counts()
        fig.add_trace(
            go.Pie(
                labels=list(data_types.index),
                values=list(data_types.values),
                name='Data Types'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Data Quality Dashboard",
            height=800,
            showlegend=True,
            template=self.config['theme']
        )
        
        # Save to HTML
        if save_path:
            fig.write_html(f"{save_path}/quality_dashboard.html")
        
        return fig
    
    def create_comprehensive_dashboard(self, data: pd.DataFrame,
                                    analysis_results: Dict[str, Any],
                                    advanced_results: Dict[str, Any] = None,
                                    quality_results: Dict[str, Any] = None,
                                    save_path: str = None) -> str:
        """
        Create a comprehensive dashboard with all components
        """
        # Create a multi-page dashboard
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Data Analytics Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .nav {{
                    background-color: #34495e;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .nav button {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    margin: 5px;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                .nav button:hover {{
                    background-color: #2980b9;
                }}
                .content {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .section {{
                    display: none;
                    margin-bottom: 30px;
                }}
                .section.active {{
                    display: block;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    text-align: center;
                    min-width: 150px;
                }}
                .metric h3 {{
                    margin: 0;
                    color: #2c3e50;
                }}
                .metric p {{
                    margin: 5px 0;
                    font-size: 24px;
                    font-weight: bold;
                    color: #27ae60;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Comprehensive Data Analytics Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="nav">
                <button onclick="showSection('overview')">📈 Overview</button>
                <button onclick="showSection('quality')">🔍 Quality Analysis</button>
                <button onclick="showSection('advanced')">🚀 Advanced Analytics</button>
                <button onclick="showSection('details')">📋 Detailed Reports</button>
            </div>
            
            <div class="content">
                <div id="overview" class="section active">
                    <h2>📈 Data Overview</h2>
                    <div class="metric">
                        <h3>Total Rows</h3>
                        <p>{len(data):,}</p>
                    </div>
                    <div class="metric">
                        <h3>Total Columns</h3>
                        <p>{len(data.columns)}</p>
                    </div>
                    <div class="metric">
                        <h3>Memory Usage</h3>
                        <p>{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB</p>
                    </div>
                    <div class="metric">
                        <h3>Quality Score</h3>
                        <p>{analysis_results.get('summary', {}).get('quality_score', 0):.1f}/100</p>
                    </div>
                </div>
                
                <div id="quality" class="section">
                    <h2>🔍 Data Quality Analysis</h2>
                    <div class="metric">
                        <h3>Missing Values</h3>
                        <p>{data.isnull().sum().sum():,}</p>
                    </div>
                    <div class="metric">
                        <h3>Duplicate Rows</h3>
                        <p>{data.duplicated().sum():,}</p>
                    </div>
                    <div class="metric">
                        <h3>Numeric Columns</h3>
                        <p>{len(data.select_dtypes(include=[np.number]).columns)}</p>
                    </div>
                    <div class="metric">
                        <h3>Categorical Columns</h3>
                        <p>{len(data.select_dtypes(include=['object']).columns)}</p>
                    </div>
                </div>
                
                <div id="advanced" class="section">
                    <h2>🚀 Advanced Analytics</h2>
                    {self._generate_advanced_section(advanced_results) if advanced_results else '<p>No advanced analytics results available.</p>'}
                </div>
                
                <div id="details" class="section">
                    <h2>📋 Detailed Reports</h2>
                    <h3>Data Types</h3>
                    <ul>
                        {self._generate_data_types_list(data)}
                    </ul>
                    
                    <h3>Missing Data by Column</h3>
                    <ul>
                        {self._generate_missing_data_list(data)}
                    </ul>
                    
                    <h3>Quality Issues</h3>
                    <ul>
                        {self._generate_quality_issues_list(analysis_results)}
                    </ul>
                </div>
            </div>
            
            <script>
                function showSection(sectionId) {{
                    // Hide all sections
                    var sections = document.getElementsByClassName('section');
                    for (var i = 0; i < sections.length; i++) {{
                        sections[i].classList.remove('active');
                    }}
                    
                    // Show selected section
                    document.getElementById(sectionId).classList.add('active');
                }}
            </script>
        </body>
        </html>
        """
        
        if save_path:
            with open(f"{save_path}/comprehensive_dashboard.html", 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
        
        return dashboard_html
    
    def _generate_advanced_section(self, advanced_results: Dict[str, Any]) -> str:
        """
        Generate HTML for advanced analytics section
        """
        if not advanced_results:
            return "<p>No advanced analytics results available.</p>"
        
        html = ""
        
        # Anomaly Detection
        if 'anomaly_detection' in advanced_results:
            anomaly_data = advanced_results['anomaly_detection']
            if 'error' not in anomaly_data:
                html += f"""
                <div class="metric">
                    <h3>Anomalies Detected</h3>
                    <p>{anomaly_data.get('anomaly_count', 0)}</p>
                </div>
                """
        
        # Clustering
        if 'clustering' in advanced_results:
            cluster_data = advanced_results['clustering']
            if 'error' not in cluster_data:
                html += f"""
                <div class="metric">
                    <h3>Clusters Found</h3>
                    <p>{cluster_data.get('n_clusters', 0)}</p>
                </div>
                """
        
        # PCA
        if 'pca_analysis' in advanced_results:
            pca_data = advanced_results['pca_analysis']
            if 'error' not in pca_data:
                html += f"""
                <div class="metric">
                    <h3>Variance Explained</h3>
                    <p>{pca_data.get('total_variance_explained', 0):.1f}%</p>
                </div>
                """
        
        return html
    
    def _generate_data_types_list(self, data: pd.DataFrame) -> str:
        """
        Generate HTML list of data types
        """
        data_types = data.dtypes
        html = ""
        for col, dtype in data_types.items():
            html += f"<li><strong>{col}:</strong> {dtype}</li>"
        return html
    
    def _generate_missing_data_list(self, data: pd.DataFrame) -> str:
        """
        Generate HTML list of missing data
        """
        missing_data = data.isnull().sum()
        html = ""
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / len(data)) * 100
                html += f"<li><strong>{col}:</strong> {missing_count} ({percentage:.1f}%)</li>"
        return html if html else "<li>No missing data found</li>"
    
    def _generate_quality_issues_list(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate HTML list of quality issues
        """
        issues = analysis_results.get('summary', {}).get('recommendations', [])
        html = ""
        for issue in issues:
            html += f"<li>{issue}</li>"
        return html if html else "<li>No quality issues found</li>"
    
    def generate_all_dashboards(self, data: pd.DataFrame,
                               analysis_results: Dict[str, Any],
                               advanced_results: Dict[str, Any] = None,
                               quality_results: Dict[str, Any] = None,
                               save_path: str = "dashboard_output") -> Dict[str, str]:
        """
        Generate all types of dashboards
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("🎨 Generating comprehensive dashboards...")
        
        dashboards = {}
        
        # Generate summary dashboard
        summary_dash = self.create_summary_dashboard(data, analysis_results, save_path)
        dashboards['summary'] = f"{save_path}/summary_dashboard.html"
        
        # Generate quality dashboard
        quality_dash = self.create_quality_dashboard(data, quality_results or analysis_results, save_path)
        dashboards['quality'] = f"{save_path}/quality_dashboard.html"
        
        # Generate advanced dashboard if results available
        if advanced_results:
            advanced_dash = self.create_advanced_dashboard(data, advanced_results, save_path)
            dashboards['advanced'] = f"{save_path}/advanced_dashboard.html"
        
        # Generate comprehensive dashboard
        comprehensive_dash = self.create_comprehensive_dashboard(
            data, analysis_results, advanced_results, quality_results, save_path
        )
        dashboards['comprehensive'] = f"{save_path}/comprehensive_dashboard.html"
        
        print(f"✅ Dashboards generated and saved to: {save_path}")
        
        return dashboards

# Example usage
def test_dashboard_generator():
    """
    Test the dashboard generator
    """
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'customer_id': range(1000),
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.lognormal(10, 0.5, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'purchase_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'product_name': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 1000),
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000),
        'is_premium': np.random.choice([True, False], 1000)
    })
    
    # Add some data quality issues
    data.loc[0:50, 'age'] = np.nan
    data.loc[100:150, 'income'] = -1000
    
    # Create sample analysis results
    analysis_results = {
        'summary': {
            'quality_score': 85.5,
            'recommendations': [
                'Some missing values detected',
                'Negative income values found',
                'Overall data quality is good'
            ]
        }
    }
    
    # Test dashboard generator
    generator = DashboardGenerator()
    dashboards = generator.generate_all_dashboards(data, analysis_results, save_path="test_dashboards")
    
    print("🎉 Dashboard generation test completed!")
    return generator, dashboards

if __name__ == "__main__":
    generator, dashboards = test_dashboard_generator() 