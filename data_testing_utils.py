import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json
from datetime import datetime
import warnings

class DataTestingUtils:
    """
    Utility functions for comprehensive data testing
    """
    
    def __init__(self):
        self.test_results = {}
    
    def test_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test data integrity and basic quality metrics
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'duplicate_rows': data.duplicated().sum(),
            'missing_data': {},
            'data_types': {},
            'constraints_violations': {},
            'anomalies': {}
        }
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        results['missing_data'] = {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': missing_data[missing_data > 0].to_dict()
        }
        
        # Data types analysis
        for col in data.columns:
            results['data_types'][col] = str(data[col].dtype)
        
        # Check for constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() == 1:
                constant_columns.append(col)
        results['constraints_violations']['constant_columns'] = constant_columns
        
        # Check for high cardinality columns
        high_cardinality = []
        for col in data.columns:
            if data[col].nunique() > len(data) * 0.8:  # More than 80% unique values
                high_cardinality.append(col)
        results['constraints_violations']['high_cardinality_columns'] = high_cardinality
        
        return results
    
    def test_numeric_data(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """
        Test numeric data for anomalies and quality issues
        """
        results = {
            'outliers': {},
            'statistics': {},
            'distribution_tests': {},
            'correlation_issues': {}
        }
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Basic statistics
            results['statistics'][col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }
            
            # Outlier detection using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            results['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(col_data)) * 100,
                'outlier_values': outliers.tolist()[:10]  # First 10 outliers
            }
            
            # Distribution tests
            results['distribution_tests'][col] = {
                'is_normal': abs(col_data.skew()) < 1 and abs(col_data.kurtosis()) < 3,
                'has_negative_values': (col_data < 0).any(),
                'has_zero_values': (col_data == 0).any()
            }
        
        # Correlation analysis
        if len(numeric_columns) > 1:
            correlation_matrix = data[numeric_columns].corr()
            high_correlations = []
            
            for i in range(len(numeric_columns)):
                for j in range(i+1, len(numeric_columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # High correlation threshold
                        high_correlations.append({
                            'column1': numeric_columns[i],
                            'column2': numeric_columns[j],
                            'correlation': corr_value
                        })
            
            results['correlation_issues']['high_correlations'] = high_correlations
        
        return results
    
    def test_categorical_data(self, data: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Any]:
        """
        Test categorical data for quality issues
        """
        results = {
            'cardinality': {},
            'value_distribution': {},
            'encoding_issues': {},
            'consistency_issues': {}
        }
        
        for col in categorical_columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Cardinality analysis
            unique_values = col_data.nunique()
            results['cardinality'][col] = {
                'unique_count': unique_values,
                'is_high_cardinality': unique_values > 50,
                'is_low_cardinality': unique_values <= 5
            }
            
            # Value distribution
            value_counts = col_data.value_counts()
            results['value_distribution'][col] = {
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
                'top_5_values': value_counts.head(5).to_dict(),
                'imbalance_ratio': value_counts.iloc[0] / value_counts.iloc[-1] if len(value_counts) > 1 else 1
            }
            
            # Check for encoding issues (mixed case, special characters)
            sample_values = col_data.head(100)
            encoding_issues = []
            
            for val in sample_values:
                if isinstance(val, str):
                    if val != val.lower() and val != val.upper():
                        encoding_issues.append('mixed_case')
                    if any(char in val for char in ['!', '@', '#', '$', '%', '^', '&', '*']):
                        encoding_issues.append('special_characters')
            
            results['encoding_issues'][col] = list(set(encoding_issues))
            
            # Consistency checks
            consistency_issues = []
            
            # Check for leading/trailing whitespace
            if col_data.dtype == 'object':
                whitespace_issues = col_data.astype(str).str.strip() != col_data.astype(str)
                if whitespace_issues.any():
                    consistency_issues.append('whitespace_issues')
            
            results['consistency_issues'][col] = consistency_issues
        
        return results
    
    def test_datetime_data(self, data: pd.DataFrame, datetime_columns: List[str]) -> Dict[str, Any]:
        """
        Test datetime data for quality issues
        """
        results = {
            'date_ranges': {},
            'format_consistency': {},
            'missing_periods': {},
            'anomalies': {}
        }
        
        for col in datetime_columns:
            col_data = pd.to_datetime(data[col], errors='coerce')
            valid_dates = col_data.dropna()
            
            if len(valid_dates) == 0:
                continue
            
            # Date range analysis
            results['date_ranges'][col] = {
                'earliest_date': valid_dates.min(),
                'latest_date': valid_dates.max(),
                'date_span_days': (valid_dates.max() - valid_dates.min()).days,
                'total_dates': len(valid_dates)
            }
            
            # Check for missing periods (gaps in time series)
            if len(valid_dates) > 1:
                sorted_dates = valid_dates.sort_values()
                date_diffs = sorted_dates.diff().dropna()
                
                if len(date_diffs) > 0:
                    median_diff = date_diffs.median()
                    large_gaps = date_diffs[date_diffs > median_diff * 3]
                    
                    results['missing_periods'][col] = {
                        'median_gap_days': median_diff.days,
                        'large_gaps_count': len(large_gaps),
                        'large_gaps_dates': large_gaps.index.tolist()[:5]  # First 5 large gaps
                    }
            
            # Format consistency
            original_col = data[col].astype(str)
            parsed_col = col_data.astype(str)
            
            format_issues = original_col != parsed_col
            results['format_consistency'][col] = {
                'format_issues_count': format_issues.sum(),
                'format_issues_percentage': (format_issues.sum() / len(original_col)) * 100
            }
            
            # Anomalies
            anomalies = []
            
            # Check for future dates (if not expected)
            future_dates = valid_dates[valid_dates > pd.Timestamp.now()]
            if len(future_dates) > 0:
                anomalies.append(f'future_dates_count: {len(future_dates)}')
            
            # Check for very old dates
            old_threshold = pd.Timestamp.now() - pd.DateOffset(years=100)
            old_dates = valid_dates[valid_dates < old_threshold]
            if len(old_dates) > 0:
                anomalies.append(f'very_old_dates_count: {len(old_dates)}')
            
            results['anomalies'][col] = anomalies
        
        return results
    
    def test_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test for data consistency across columns
        """
        results = {
            'cross_column_validation': {},
            'business_rules': {},
            'referential_integrity': {}
        }
        
        # Example cross-column validations
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Check for logical inconsistencies
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                # Check if one column should always be greater than another
                if 'total' in col1.lower() and 'subtotal' in col2.lower():
                    invalid_rows = data[data[col1] < data[col2]]
                    if len(invalid_rows) > 0:
                        results['cross_column_validation'][f'{col1}_vs_{col2}'] = {
                            'issue': 'total_less_than_subtotal',
                            'invalid_rows_count': len(invalid_rows)
                        }
        
        # Check for business rules (example)
        if 'age' in data.columns and 'birth_date' in data.columns:
            try:
                birth_dates = pd.to_datetime(data['birth_date'], errors='coerce')
                calculated_ages = (pd.Timestamp.now() - birth_dates).dt.total_seconds() / (365.25 * 24 * 3600)
                age_discrepancies = abs(data['age'] - calculated_ages) > 1
                
                if age_discrepancies.any():
                    results['business_rules']['age_birth_date_mismatch'] = {
                        'discrepancies_count': age_discrepancies.sum()
                    }
            except:
                pass
        
        return results
    
    def generate_test_report(self, data: pd.DataFrame, save_path: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive test report
        """
        print("🧪 Starting comprehensive data testing...")
        
        # Categorize columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Run all tests
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'data_integrity': self.test_data_integrity(data),
            'numeric_tests': self.test_numeric_data(data, numeric_columns),
            'categorical_tests': self.test_categorical_data(data, categorical_columns),
            'datetime_tests': self.test_datetime_data(data, datetime_columns),
            'consistency_tests': self.test_data_consistency(data),
            'summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'warnings': 0,
                'recommendations': []
            }
        }
        
        # Generate summary and recommendations
        issues = []
        
        # Check for critical issues
        if test_results['data_integrity']['missing_data']['missing_percentage'] > 50:
            issues.append("CRITICAL: More than 50% of data is missing")
            test_results['summary']['critical_issues'] += 1
        
        if test_results['data_integrity']['duplicate_rows'] > len(data) * 0.1:
            issues.append("CRITICAL: More than 10% of rows are duplicates")
            test_results['summary']['critical_issues'] += 1
        
        # Check for warnings
        for col, outlier_info in test_results['numeric_tests']['outliers'].items():
            if outlier_info['percentage'] > 10:
                issues.append(f"WARNING: Column '{col}' has more than 10% outliers")
                test_results['summary']['warnings'] += 1
        
        for col, cardinality_info in test_results['categorical_tests']['cardinality'].items():
            if cardinality_info['is_high_cardinality']:
                issues.append(f"WARNING: Column '{col}' has high cardinality (>50 unique values)")
                test_results['summary']['warnings'] += 1
        
        test_results['summary']['total_issues'] = test_results['summary']['critical_issues'] + test_results['summary']['warnings']
        test_results['summary']['recommendations'] = issues
        
        # Save report if path provided
        if save_path:
            with open(f"{save_path}/data_test_report.json", 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
        
        print(f"✅ Data testing complete! Found {test_results['summary']['total_issues']} issues")
        
        return test_results
    
    def validate_data_schema(self, data: pd.DataFrame, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against an expected schema
        """
        results = {
            'schema_validation': {},
            'column_validation': {},
            'data_type_validation': {},
            'constraint_validation': {}
        }
        
        # Check if all expected columns exist
        expected_columns = expected_schema.get('columns', [])
        missing_columns = set(expected_columns) - set(data.columns)
        extra_columns = set(data.columns) - set(expected_columns)
        
        results['schema_validation'] = {
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'schema_match': len(missing_columns) == 0
        }
        
        # Validate data types
        expected_types = expected_schema.get('data_types', {})
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                results['data_type_validation'][col] = {
                    'expected': expected_type,
                    'actual': actual_type,
                    'match': expected_type in actual_type or actual_type in expected_type
                }
        
        # Validate constraints
        constraints = expected_schema.get('constraints', {})
        for col, constraint in constraints.items():
            if col in data.columns:
                validation_result = self._validate_constraint(data[col], constraint)
                results['constraint_validation'][col] = validation_result
        
        return results
    
    def _validate_constraint(self, column_data: pd.Series, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single column against constraints
        """
        result = {
            'constraint_type': constraint.get('type'),
            'valid': True,
            'violations': 0,
            'details': {}
        }
        
        constraint_type = constraint.get('type')
        
        if constraint_type == 'range':
            min_val = constraint.get('min')
            max_val = constraint.get('max')
            
            if min_val is not None:
                violations = column_data < min_val
                result['details']['below_min'] = violations.sum()
                result['valid'] = result['valid'] and violations.sum() == 0
            
            if max_val is not None:
                violations = column_data > max_val
                result['details']['above_max'] = violations.sum()
                result['valid'] = result['valid'] and violations.sum() == 0
        
        elif constraint_type == 'unique':
            duplicates = column_data.duplicated()
            result['details']['duplicates'] = duplicates.sum()
            result['valid'] = result['valid'] and duplicates.sum() == 0
        
        elif constraint_type == 'not_null':
            null_count = column_data.isnull().sum()
            result['details']['null_count'] = null_count
            result['valid'] = result['valid'] and null_count == 0
        
        elif constraint_type == 'enum':
            allowed_values = constraint.get('values', [])
            invalid_values = ~column_data.isin(allowed_values)
            result['details']['invalid_values'] = invalid_values.sum()
            result['valid'] = result['valid'] and invalid_values.sum() == 0
        
        result['violations'] = sum(result['details'].values())
        
        return result

# Example usage
def test_data_testing_utils():
    """
    Test the data testing utilities
    """
    # Create sample data with various issues
    sample_data = pd.DataFrame({
        'id': range(1000),
        'age': np.random.normal(35, 10, 1000),
        'salary': np.random.lognormal(10, 0.5, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'text': [f'text_{i}' for i in range(1000)]
    })
    
    # Add some data quality issues
    sample_data.loc[0:50, 'age'] = np.nan  # Missing values
    sample_data.loc[100:150, 'salary'] = -1000  # Negative salary
    sample_data.loc[200:250, 'age'] = 200  # Impossible age
    sample_data.loc[300:350, 'category'] = 'INVALID'  # Invalid category
    
    # Test the utilities
    utils = DataTestingUtils()
    test_report = utils.generate_test_report(sample_data, "test_output")
    
    print("🎉 Data testing utilities test completed!")
    return utils, test_report

if __name__ == "__main__":
    utils, report = test_data_testing_utils() 