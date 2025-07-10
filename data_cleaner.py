#!/usr/bin/env python3
"""
Data Cleaning and Preprocessing Module
Automated data cleaning with multiple strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing
    """
    
    def __init__(self):
        self.cleaning_log = []
        self.cleaning_stats = {}
        self.original_data = None
        self.cleaned_data = None
    
    def log_cleaning_action(self, action: str, details: Dict[str, Any]):
        """
        Log cleaning actions for transparency
        """
        self.cleaning_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })
    
    def detect_data_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Enhanced data type detection
        """
        data_types = {}
        
        for column in data.columns:
            # Check for datetime columns
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data_types[column] = 'datetime'
            # Check for boolean columns
            elif pd.api.types.is_bool_dtype(data[column]):
                data_types[column] = 'boolean'
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(data[column]):
                if data[column].nunique() <= 10:
                    data_types[column] = 'categorical_numeric'
                else:
                    data_types[column] = 'numeric'
            # Check for categorical/text columns
            elif pd.api.types.is_string_dtype(data[column]) or pd.api.types.is_object_dtype(data[column]):
                if data[column].nunique() <= 50:
                    data_types[column] = 'categorical'
                else:
                    data_types[column] = 'text'
            else:
                data_types[column] = 'unknown'
        
        return data_types
    
    def clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names
        """
        cleaned_data = data.copy()
        
        # Remove special characters and spaces
        cleaned_columns = {}
        for col in cleaned_data.columns:
            # Convert to string if not already
            col_str = str(col)
            
            # Remove special characters except underscores
            cleaned_col = re.sub(r'[^a-zA-Z0-9_]', '_', col_str)
            
            # Remove multiple underscores
            cleaned_col = re.sub(r'_+', '_', cleaned_col)
            
            # Remove leading/trailing underscores
            cleaned_col = cleaned_col.strip('_')
            
            # Convert to lowercase
            cleaned_col = cleaned_col.lower()
            
            # Ensure unique column names
            if cleaned_col in cleaned_columns.values():
                counter = 1
                original_col = cleaned_col
                while cleaned_col in cleaned_columns.values():
                    cleaned_col = f"{original_col}_{counter}"
                    counter += 1
            
            cleaned_columns[col] = cleaned_col
        
        # Rename columns
        cleaned_data = cleaned_data.rename(columns=cleaned_columns)
        
        self.log_cleaning_action('clean_column_names', {
            'original_columns': list(data.columns),
            'cleaned_columns': list(cleaned_data.columns),
            'changes_made': len([k for k, v in cleaned_columns.items() if k != v])
        })
        
        return cleaned_data
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values with various strategies
        """
        cleaned_data = data.copy()
        missing_stats = data.isnull().sum()
        
        if strategy == 'auto':
            # Auto-detect best strategy for each column
            for column in data.columns:
                missing_count = missing_stats[column]
                total_count = len(data)
                missing_percentage = (missing_count / total_count) * 100
                
                if missing_percentage > 50:
                    # More than 50% missing - drop column
                    cleaned_data = cleaned_data.drop(columns=[column])
                    self.log_cleaning_action('drop_column_high_missing', {
                        'column': column,
                        'missing_percentage': missing_percentage
                    })
                elif missing_percentage > 10:
                    # 10-50% missing - use median/mean for numeric, mode for categorical
                    if pd.api.types.is_numeric_dtype(data[column]):
                        fill_value = data[column].median()
                        cleaned_data[column] = cleaned_data[column].fillna(fill_value)
                        self.log_cleaning_action('fill_numeric_missing', {
                            'column': column,
                            'method': 'median',
                            'fill_value': fill_value
                        })
                    else:
                        fill_value = data[column].mode().iloc[0] if not data[column].mode().empty else 'Unknown'
                        cleaned_data[column] = cleaned_data[column].fillna(fill_value)
                        self.log_cleaning_action('fill_categorical_missing', {
                            'column': column,
                            'method': 'mode',
                            'fill_value': fill_value
                        })
                else:
                    # Less than 10% missing - drop rows
                    cleaned_data = cleaned_data.dropna(subset=[column])
                    self.log_cleaning_action('drop_rows_missing', {
                        'column': column,
                        'rows_dropped': missing_count
                    })
        
        elif strategy == 'drop':
            # Drop all rows with missing values
            original_len = len(cleaned_data)
            cleaned_data = cleaned_data.dropna()
            dropped_rows = original_len - len(cleaned_data)
            self.log_cleaning_action('drop_all_missing', {
                'rows_dropped': dropped_rows
            })
        
        elif strategy == 'fill':
            # Fill missing values
            for column in data.columns:
                if missing_stats[column] > 0:
                    if pd.api.types.is_numeric_dtype(data[column]):
                        fill_value = data[column].median()
                        cleaned_data[column] = cleaned_data[column].fillna(fill_value)
                    else:
                        fill_value = data[column].mode().iloc[0] if not data[column].mode().empty else 'Unknown'
                        cleaned_data[column] = cleaned_data[column].fillna(fill_value)
        
        return cleaned_data
    
    def remove_duplicates(self, data: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows
        """
        cleaned_data = data.copy()
        original_len = len(cleaned_data)
        
        cleaned_data = cleaned_data.drop_duplicates(subset=subset)
        removed_duplicates = original_len - len(cleaned_data)
        
        self.log_cleaning_action('remove_duplicates', {
            'original_rows': original_len,
            'cleaned_rows': len(cleaned_data),
            'duplicates_removed': removed_duplicates,
            'subset_used': subset
        })
        
        return cleaned_data
    
    def clean_outliers(self, data: pd.DataFrame, method: str = 'iqr', 
                      columns: List[str] = None) -> pd.DataFrame:
        """
        Clean outliers using various methods
        """
        cleaned_data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_removed = 0
        
        for column in columns:
            if column not in data.columns:
                continue
            
            col_data = data[column].dropna()
            if len(col_data) == 0:
                continue
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                outlier_mask = (cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)
                outliers_count = outlier_mask.sum()
                cleaned_data = cleaned_data[~outlier_mask]
                outliers_removed += outliers_count
                
                self.log_cleaning_action('remove_outliers_iqr', {
                    'column': column,
                    'outliers_removed': outliers_count,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
            
            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outlier_mask = z_scores > 3
                outliers_count = outlier_mask.sum()
                
                # Apply to full dataset
                full_outlier_mask = (np.abs((cleaned_data[column] - col_data.mean()) / col_data.std()) > 3)
                cleaned_data = cleaned_data[~full_outlier_mask]
                outliers_removed += outliers_count
                
                self.log_cleaning_action('remove_outliers_zscore', {
                    'column': column,
                    'outliers_removed': outliers_count,
                    'threshold': 3
                })
        
        return cleaned_data
    
    def standardize_text_columns(self, data: pd.DataFrame, 
                               columns: List[str] = None) -> pd.DataFrame:
        """
        Standardize text columns
        """
        cleaned_data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=['object']).columns.tolist()
        
        for column in columns:
            if column not in data.columns:
                continue
            
            # Convert to string
            cleaned_data[column] = cleaned_data[column].astype(str)
            
            # Remove leading/trailing whitespace
            cleaned_data[column] = cleaned_data[column].str.strip()
            
            # Convert to lowercase
            cleaned_data[column] = cleaned_data[column].str.lower()
            
            # Replace multiple spaces with single space
            cleaned_data[column] = cleaned_data[column].str.replace(r'\s+', ' ', regex=True)
            
            # Handle empty strings
            cleaned_data[column] = cleaned_data[column].replace(['', 'nan', 'none'], np.nan)
        
        self.log_cleaning_action('standardize_text', {
            'columns_processed': columns
        })
        
        return cleaned_data
    
    def convert_data_types(self, data: pd.DataFrame, 
                          type_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Convert data types based on content analysis
        """
        cleaned_data = data.copy()
        
        if type_mapping is None:
            # Auto-detect and convert types
            for column in cleaned_data.columns:
                # Try to convert to numeric
                try:
                    pd.to_numeric(cleaned_data[column], errors='raise')
                    cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors='coerce')
                    self.log_cleaning_action('convert_to_numeric', {
                        'column': column,
                        'success': True
                    })
                except:
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(cleaned_data[column], errors='raise')
                        cleaned_data[column] = pd.to_datetime(cleaned_data[column], errors='coerce')
                        self.log_cleaning_action('convert_to_datetime', {
                            'column': column,
                            'success': True
                        })
                    except:
                        # Keep as object/string
                        pass
        else:
            # Use provided type mapping
            for column, target_type in type_mapping.items():
                if column in cleaned_data.columns:
                    try:
                        if target_type == 'numeric':
                            cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors='coerce')
                        elif target_type == 'datetime':
                            cleaned_data[column] = pd.to_datetime(cleaned_data[column], errors='coerce')
                        elif target_type == 'category':
                            cleaned_data[column] = cleaned_data[column].astype('category')
                        
                        self.log_cleaning_action('convert_data_type', {
                            'column': column,
                            'target_type': target_type,
                            'success': True
                        })
                    except Exception as e:
                        self.log_cleaning_action('convert_data_type', {
                            'column': column,
                            'target_type': target_type,
                            'success': False,
                            'error': str(e)
                        })
        
        return cleaned_data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality after cleaning
        """
        quality_report = {
            'rows': len(data),
            'columns': len(data.columns),
            'missing_data': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': self.detect_data_types(data),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'quality_score': 0
        }
        
        # Calculate quality score
        score = 100
        
        # Deduct for missing data
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        score -= missing_percentage * 0.5
        
        # Deduct for duplicates
        duplicate_percentage = (data.duplicated().sum() / len(data)) * 100
        score -= duplicate_percentage * 0.3
        
        # Deduct for unknown data types
        unknown_types = sum(1 for dt in quality_report['data_types'].values() if dt == 'unknown')
        score -= unknown_types * 5
        
        quality_report['quality_score'] = max(0, score)
        
        return quality_report
    
    def comprehensive_clean(self, data: pd.DataFrame, 
                          cleaning_options: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive data cleaning pipeline
        """
        if cleaning_options is None:
            cleaning_options = {
                'clean_column_names': True,
                'handle_missing_values': True,
                'remove_duplicates': True,
                'clean_outliers': True,
                'standardize_text': True,
                'convert_data_types': True,
                'missing_strategy': 'auto',
                'outlier_method': 'iqr'
            }
        
        self.original_data = data.copy()
        cleaned_data = data.copy()
        
        print("🧹 Starting comprehensive data cleaning...")
        
        # Step 1: Clean column names
        if cleaning_options.get('clean_column_names', True):
            print("  📝 Cleaning column names...")
            cleaned_data = self.clean_column_names(cleaned_data)
        
        # Step 2: Handle missing values
        if cleaning_options.get('handle_missing_values', True):
            print("  🔍 Handling missing values...")
            strategy = cleaning_options.get('missing_strategy', 'auto')
            cleaned_data = self.handle_missing_values(cleaned_data, strategy)
        
        # Step 3: Remove duplicates
        if cleaning_options.get('remove_duplicates', True):
            print("  🗑️ Removing duplicates...")
            cleaned_data = self.remove_duplicates(cleaned_data)
        
        # Step 4: Clean outliers
        if cleaning_options.get('clean_outliers', True):
            print("  📊 Cleaning outliers...")
            method = cleaning_options.get('outlier_method', 'iqr')
            cleaned_data = self.clean_outliers(cleaned_data, method)
        
        # Step 5: Standardize text
        if cleaning_options.get('standardize_text', True):
            print("  📝 Standardizing text...")
            cleaned_data = self.standardize_text_columns(cleaned_data)
        
        # Step 6: Convert data types
        if cleaning_options.get('convert_data_types', True):
            print("  🔄 Converting data types...")
            cleaned_data = self.convert_data_types(cleaned_data)
        
        # Generate cleaning statistics
        self.cleaning_stats = {
            'original_shape': self.original_data.shape,
            'cleaned_shape': cleaned_data.shape,
            'rows_removed': self.original_data.shape[0] - cleaned_data.shape[0],
            'columns_removed': self.original_data.shape[1] - cleaned_data.shape[1],
            'cleaning_log': self.cleaning_log,
            'quality_report': self.validate_data_quality(cleaned_data)
        }
        
        self.cleaned_data = cleaned_data
        
        print(f"✅ Data cleaning complete!")
        print(f"  Original shape: {self.original_data.shape}")
        print(f"  Cleaned shape: {cleaned_data.shape}")
        print(f"  Quality score: {self.cleaning_stats['quality_report']['quality_score']:.1f}/100")
        
        return cleaned_data, self.cleaning_stats
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the cleaning process
        """
        if self.cleaning_stats is None:
            return {"error": "No cleaning has been performed yet"}
        
        summary = {
            'cleaning_summary': {
                'original_rows': self.cleaning_stats['original_shape'][0],
                'cleaned_rows': self.cleaning_stats['cleaned_shape'][0],
                'rows_removed': self.cleaning_stats['rows_removed'],
                'original_columns': self.cleaning_stats['original_shape'][1],
                'cleaned_columns': self.cleaning_stats['cleaned_shape'][1],
                'columns_removed': self.cleaning_stats['columns_removed']
            },
            'quality_score': self.cleaning_stats['quality_report']['quality_score'],
            'actions_performed': len(self.cleaning_log),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on cleaning results
        """
        recommendations = []
        
        if self.cleaning_stats is None:
            return recommendations
        
        quality_score = self.cleaning_stats['quality_report']['quality_score']
        
        if quality_score < 50:
            recommendations.append("Data quality is poor. Consider manual review of data sources.")
        elif quality_score < 70:
            recommendations.append("Data quality is moderate. Some manual cleaning may be needed.")
        elif quality_score < 90:
            recommendations.append("Data quality is good. Minor improvements possible.")
        else:
            recommendations.append("Data quality is excellent!")
        
        if self.cleaning_stats['rows_removed'] > 0:
            recommendations.append(f"Removed {self.cleaning_stats['rows_removed']} rows during cleaning.")
        
        if self.cleaning_stats['columns_removed'] > 0:
            recommendations.append(f"Removed {self.cleaning_stats['columns_removed']} columns during cleaning.")
        
        return recommendations

# Example usage
def test_data_cleaner():
    """
    Test the data cleaner with sample data
    """
    # Create sample data with various issues
    data = pd.DataFrame({
        'Customer ID': range(1000),
        'Age': np.random.normal(35, 10, 1000),
        'Income': np.random.lognormal(10, 0.5, 1000),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'Purchase Date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'Product Name': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 1000),
        'Rating': np.random.choice([1, 2, 3, 4, 5], 1000),
        'Is Premium': np.random.choice([True, False], 1000),
        'Notes': [f'Note for customer {i}' for i in range(1000)]
    })
    
    # Add some data quality issues
    data.loc[0:50, 'Age'] = np.nan  # Missing values
    data.loc[100:150, 'Income'] = -1000  # Negative income
    data.loc[200:250, 'Age'] = 200  # Impossible age
    data.loc[300:350, 'Category'] = 'INVALID'  # Invalid category
    data.loc[400:450, 'Rating'] = 10  # Invalid rating
    data.loc[500:550, 'Purchase Date'] = pd.NaT  # Missing dates
    
    # Add some duplicates
    data = pd.concat([data, data.iloc[0:10]], ignore_index=True)
    
    # Add some text issues
    data.loc[600:650, 'Notes'] = '  MULTIPLE   SPACES  '
    data.loc[700:750, 'Notes'] = 'UPPERCASE TEXT'
    
    # Test the cleaner
    cleaner = DataCleaner()
    cleaned_data, stats = cleaner.comprehensive_clean(data)
    
    print("\n📊 Cleaning Summary:")
    summary = cleaner.get_cleaning_summary()
    for key, value in summary['cleaning_summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\n🎯 Quality Score: {summary['quality_score']:.1f}/100")
    
    if summary['recommendations']:
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    return cleaner, cleaned_data, stats

if __name__ == "__main__":
    cleaner, cleaned_data, stats = test_data_cleaner() 