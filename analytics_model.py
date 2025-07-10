import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import json
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ydata_profiling
from ydata_profiling import ProfileReport
import os

class DataAnalyticsModel:
    """
    Comprehensive Analytics Model for Data Testing
    Can handle any kind of data with multiple analysis types
    """
    
    def __init__(self, data_source: Union[str, pd.DataFrame] = None):
        """
        Initialize the analytics model
        
        Args:
            data_source: Path to data file or pandas DataFrame
        """
        self.data = None
        self.analysis_results = {}
        self.data_quality_report = {}
        self.visualizations = {}
        
        if data_source is not None:
            self.load_data(data_source)
    
    def load_data(self, data_source: Union[str, pd.DataFrame]) -> None:
        """
        Load data from various sources
        
        Args:
            data_source: File path or pandas DataFrame
        """
        try:
            if isinstance(data_source, str):
                # Try different file formats
                if data_source.endswith('.csv'):
                    self.data = pd.read_csv(data_source)
                elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                    self.data = pd.read_excel(data_source)
                elif data_source.endswith('.json'):
                    self.data = pd.read_json(data_source)
                elif data_source.endswith('.parquet'):
                    self.data = pd.read_parquet(data_source)
                else:
                    raise ValueError(f"Unsupported file format: {data_source}")
            elif isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
            else:
                raise ValueError("data_source must be a file path or pandas DataFrame")
            
            print(f"✅ Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def detect_data_types(self) -> Dict[str, str]:
        """
        Automatically detect and categorize data types
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        data_types = {}
        for column in self.data.columns:
            # Check for datetime columns
            if pd.api.types.is_datetime64_any_dtype(self.data[column]):
                data_types[column] = 'datetime'
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                if self.data[column].nunique() <= 10:
                    data_types[column] = 'categorical_numeric'
                else:
                    data_types[column] = 'numeric'
            # Check for categorical/text columns
            elif pd.api.types.is_string_dtype(self.data[column]) or pd.api.types.is_object_dtype(self.data[column]):
                if self.data[column].nunique() <= 50:
                    data_types[column] = 'categorical'
                else:
                    data_types[column] = 'text'
            else:
                data_types[column] = 'unknown'
        
        return data_types
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        quality_report = {
            'basic_info': {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'duplicate_rows': self.data.duplicated().sum()
            },
            'missing_data': {},
            'data_types': self.detect_data_types(),
            'outliers': {},
            'statistics': {}
        }
        
        # Missing data analysis
        missing_data = self.data.isnull().sum()
        quality_report['missing_data'] = {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(self.data) * len(self.data.columns))) * 100,
            'columns_with_missing': missing_data[missing_data > 0].to_dict()
        }
        
        # Outlier detection for numeric columns
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            quality_report['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.data)) * 100
            }
        
        # Basic statistics
        quality_report['statistics'] = {
            'numeric_stats': self.data.describe().to_dict() if len(numeric_columns) > 0 else {},
            'categorical_stats': {}
        }
        
        # Categorical statistics
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            quality_report['statistics']['categorical_stats'][col] = {
                'unique_values': self.data[col].nunique(),
                'most_common': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                'value_counts': self.data[col].value_counts().head(10).to_dict()
            }
        
        self.data_quality_report = quality_report
        return quality_report
    
    def generate_visualizations(self, save_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations for the data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        viz_results = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Missing data heatmap
        plt.figure(figsize=(12, 6))
        missing_data = self.data.isnull()
        sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Data Heatmap')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/missing_data_heatmap.png", dpi=300, bbox_inches='tight')
        viz_results['missing_heatmap'] = plt.gcf()
        plt.close()
        
        # 2. Data types distribution
        data_types = self.detect_data_types()
        type_counts = pd.Series(list(data_types.values())).value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        type_counts.plot(kind='bar', ax=ax)
        plt.title('Data Types Distribution')
        plt.xlabel('Data Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/data_types_distribution.png", dpi=300, bbox_inches='tight')
        viz_results['data_types'] = fig
        plt.close()
        
        # 3. Numeric columns analysis
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.data[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            viz_results['correlation'] = plt.gcf()
            plt.close()
            
            # Distribution plots for numeric columns
            n_numeric = len(numeric_columns)
            cols_per_row = 3
            rows = (n_numeric + cols_per_row - 1) // cols_per_row
            
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(numeric_columns):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                ax = axes[row, col_idx]
                
                self.data[col].hist(ax=ax, bins=30, alpha=0.7)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_columns), rows * cols_per_row):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/numeric_distributions.png", dpi=300, bbox_inches='tight')
            viz_results['numeric_distributions'] = fig
            plt.close()
        
        # 4. Categorical columns analysis
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            n_categorical = len(categorical_columns)
            cols_per_row = 2
            rows = (n_categorical + cols_per_row - 1) // cols_per_row
            
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(categorical_columns):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                ax = axes[row, col_idx]
                
                value_counts = self.data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Top 10 Values in {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Hide empty subplots
            for i in range(len(categorical_columns), rows * cols_per_row):
                row = i // cols_per_row
                col_idx = i % cols_per_row
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/categorical_analysis.png", dpi=300, bbox_inches='tight')
            viz_results['categorical_analysis'] = fig
            plt.close()
        
        self.visualizations = viz_results
        return viz_results
    
    def generate_interactive_dashboard(self, save_path: str = None) -> None:
        """
        Generate an interactive Plotly dashboard
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create interactive visualizations
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        # Create subplots
        n_plots = 0
        if len(numeric_columns) > 0:
            n_plots += 1
        if len(categorical_columns) > 0:
            n_plots += 1
        if len(numeric_columns) > 1:
            n_plots += 1
        
        if n_plots == 0:
            print("No suitable columns for interactive visualization")
            return
        
        fig = make_subplots(
            rows=n_plots, cols=1,
            subplot_titles=['Numeric Distributions', 'Categorical Analysis', 'Correlation Matrix'][:n_plots],
            vertical_spacing=0.1
        )
        
        plot_idx = 1
        
        # Numeric distributions
        if len(numeric_columns) > 0:
            for col in numeric_columns:
                fig.add_trace(
                    go.Histogram(x=self.data[col], name=col, opacity=0.7),
                    row=plot_idx, col=1
                )
            plot_idx += 1
        
        # Categorical analysis
        if len(categorical_columns) > 0:
            for col in categorical_columns[:3]:  # Limit to first 3 categorical columns
                value_counts = self.data[col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name=col),
                    row=plot_idx, col=1
                )
            plot_idx += 1
        
        # Correlation matrix
        if len(numeric_columns) > 1:
            correlation_matrix = self.data[numeric_columns].corr()
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=plot_idx, col=1
            )
        
        fig.update_layout(
            height=300 * n_plots,
            title_text="Interactive Data Analysis Dashboard",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(f"{save_path}/interactive_dashboard.html")
        
        return fig
    
    def generate_profiling_report(self, save_path: str = None) -> ProfileReport:
        """
        Generate a comprehensive profiling report using ydata-profiling
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        try:
            profile = ProfileReport(
                self.data,
                title="Data Analytics Report",
                explorative=True,
                dark_mode=False,
                html={'style': {'full_width': True}}
            )
            
            if save_path:
                profile.to_file(f"{save_path}/profiling_report.html")
            
            return profile
            
        except Exception as e:
            print(f"Error generating profiling report: {str(e)}")
            return None
    
    def run_comprehensive_analysis(self, save_path: str = "analytics_output") -> Dict[str, Any]:
        """
        Run comprehensive analysis including all components
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        print("🔍 Starting comprehensive data analysis...")
        
        # 1. Data quality analysis
        print("📊 Analyzing data quality...")
        quality_report = self.analyze_data_quality()
        
        # 2. Generate visualizations
        print("📈 Generating visualizations...")
        viz_results = self.generate_visualizations(save_path)
        
        # 3. Generate interactive dashboard
        print("🎛️ Creating interactive dashboard...")
        dashboard = self.generate_interactive_dashboard(save_path)
        
        # 4. Generate profiling report
        print("📋 Generating profiling report...")
        profiling_report = self.generate_profiling_report(save_path)
        
        # 5. Save analysis results
        analysis_summary = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': self.data.shape,
            'data_types': self.detect_data_types(),
            'quality_report': quality_report,
            'analysis_complete': True
        }
        
        with open(f"{save_path}/analysis_summary.json", 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        print(f"✅ Analysis complete! Results saved to: {save_path}")
        
        return {
            'quality_report': quality_report,
            'visualizations': viz_results,
            'dashboard': dashboard,
            'profiling_report': profiling_report,
            'summary': analysis_summary
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a quick summary of the data
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data_types': self.detect_data_types(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'duplicates': self.data.duplicated().sum()
        }

# Example usage and testing
def test_analytics_model():
    """
    Test the analytics model with sample data
    """
    # Create sample data
    sample_data = pd.DataFrame({
        'numeric_col': np.random.normal(0, 1, 1000),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 1000),
        'text_col': [f'text_{i}' for i in range(1000)],
        'date_col': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'mixed_col': np.random.choice([1, 2, 3, 'text'], 1000)
    })
    
    # Add some missing values
    sample_data.loc[0:50, 'numeric_col'] = np.nan
    sample_data.loc[100:150, 'categorical_col'] = np.nan
    
    # Initialize and test the model
    model = DataAnalyticsModel(sample_data)
    
    # Run comprehensive analysis
    results = model.run_comprehensive_analysis("test_output")
    
    print("🎉 Analytics model test completed successfully!")
    return model, results

if __name__ == "__main__":
    # Test the model
    model, results = test_analytics_model() 