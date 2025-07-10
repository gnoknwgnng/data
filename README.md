# Comprehensive Data Analytics & Testing System

A powerful analytics system that can handle any kind of data with comprehensive testing, visualization, and quality assessment capabilities.

## 🚀 Features

### Core Analytics Capabilities
- **Universal Data Support**: Handles CSV, Excel, JSON, Parquet files
- **Automatic Data Type Detection**: Intelligently categorizes numeric, categorical, datetime, and text data
- **Comprehensive Quality Analysis**: Missing data, outliers, duplicates, and consistency checks
- **Interactive Visualizations**: Static plots and interactive Plotly dashboards
- **Profiling Reports**: Detailed HTML reports with ydata-profiling

### Data Testing Features
- **Data Integrity Testing**: Validates data structure and completeness
- **Numeric Data Analysis**: Outlier detection, distribution analysis, correlation testing
- **Categorical Data Testing**: Cardinality analysis, value distribution, encoding issues
- **Datetime Validation**: Date range analysis, format consistency, anomaly detection
- **Schema Validation**: Custom constraint testing and business rule validation

### Quality Scoring
- **Overall Quality Score**: 0-100 score based on multiple factors
- **Issue Classification**: Critical issues vs warnings
- **Actionable Recommendations**: Specific suggestions for data improvement

## 📦 Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### 1. Create Sample Data (for testing)
```bash
python main_analytics.py --create-sample
```

### 2. Run Quick Analysis
```bash
python main_analytics.py --data sample_data.csv --quick
```

### 3. Run Full Analysis
```bash
python main_analytics.py --data your_data.csv --output results
```

## 📊 Usage Examples

### Basic Analysis
```python
from main_analytics import ComprehensiveDataAnalytics

# Initialize the system
analytics = ComprehensiveDataAnalytics()

# Run comprehensive analysis
results = analytics.analyze_data("your_data.csv", "output_directory")
```

### Quick Analysis
```python
# Get quick insights without generating plots
results = analytics.quick_analysis("your_data.csv")
print(f"Quality Score: {results['summary']['quality_score']}")
```

### Using Individual Components
```python
from analytics_model import DataAnalyticsModel
from data_testing_utils import DataTestingUtils

# Analytics model
model = DataAnalyticsModel("your_data.csv")
quality_report = model.analyze_data_quality()
visualizations = model.generate_visualizations()

# Testing utilities
utils = DataTestingUtils()
test_results = utils.generate_test_report(model.data)
```

## 🔧 Command Line Options

```bash
python main_analytics.py [OPTIONS]

Options:
  --data, -d PATH          Path to data file
  --output, -o DIR         Output directory (default: analytics_output)
  --quick, -q              Quick analysis only (no plots)
  --create-sample          Create sample data for testing
  --no-plots               Skip generating static plots
  --no-dashboard           Skip generating interactive dashboard
  --no-profiling           Skip generating profiling report
```

## 📈 Output Files

The system generates several output files:

### Analysis Results
- `combined_results.json` - Complete analysis results
- `analysis_summary.txt` - Human-readable summary
- `data_test_report.json` - Detailed testing results

### Visualizations
- `missing_data_heatmap.png` - Missing data visualization
- `data_types_distribution.png` - Data type distribution
- `correlation_matrix.png` - Numeric correlations
- `numeric_distributions.png` - Distribution plots
- `categorical_analysis.png` - Categorical data analysis

### Interactive Reports
- `interactive_dashboard.html` - Plotly interactive dashboard
- `profiling_report.html` - Comprehensive ydata-profiling report

## 🧪 Data Quality Tests

### Automatic Tests Performed
1. **Data Integrity**
   - Missing data analysis
   - Duplicate row detection
   - Data type validation
   - Constant column detection

2. **Numeric Data**
   - Outlier detection (IQR method)
   - Distribution analysis
   - Correlation testing
   - Statistical summaries

3. **Categorical Data**
   - Cardinality analysis
   - Value distribution
   - Encoding consistency
   - Format validation

4. **Datetime Data**
   - Date range validation
   - Format consistency
   - Missing periods detection
   - Anomaly detection

5. **Cross-Column Validation**
   - Business rule testing
   - Logical consistency
   - Referential integrity

## 📊 Quality Scoring System

The system calculates a quality score (0-100) based on:

- **Missing Data**: -0.5 points per 1% missing data
- **Duplicate Rows**: -0.3 points per 1% duplicates
- **Critical Issues**: -10 points per critical issue
- **Warnings**: -2 points per warning

## 🔍 Understanding Results

### Quality Score Interpretation
- **90-100**: Excellent data quality
- **70-89**: Good data quality with minor issues
- **50-69**: Moderate data quality, needs attention
- **30-49**: Poor data quality, significant issues
- **0-29**: Very poor data quality, major problems

### Common Issues and Solutions

#### Missing Data
- **Issue**: High percentage of missing values
- **Solution**: Impute missing values or remove affected rows/columns

#### Outliers
- **Issue**: Extreme values in numeric columns
- **Solution**: Investigate outliers, consider removal or transformation

#### High Cardinality
- **Issue**: Too many unique values in categorical columns
- **Solution**: Group similar values or use encoding techniques

#### Duplicate Rows
- **Issue**: Repeated data entries
- **Solution**: Remove duplicates or investigate data collection process

## 🛠️ Customization

### Adding Custom Tests
```python
class CustomDataTestingUtils(DataTestingUtils):
    def test_custom_business_rules(self, data):
        # Add your custom business logic here
        pass
```

### Custom Schema Validation
```python
expected_schema = {
    'columns': ['id', 'name', 'age', 'email'],
    'data_types': {
        'id': 'int64',
        'name': 'object',
        'age': 'int64',
        'email': 'object'
    },
    'constraints': {
        'age': {'type': 'range', 'min': 0, 'max': 120},
        'email': {'type': 'not_null'}
    }
}

validation_results = utils.validate_data_schema(data, expected_schema)
```

## 🤝 Contributing

To extend the system:

1. **Add new test types** in `data_testing_utils.py`
2. **Enhance visualizations** in `analytics_model.py`
3. **Create custom analyzers** by extending the base classes
4. **Add new data formats** by updating the load_data method

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: For large datasets, use quick analysis
   ```bash
   python main_analytics.py --data large_file.csv --quick
   ```

3. **Plotting Errors**: Install additional dependencies
   ```bash
   pip install matplotlib seaborn plotly
   ```

4. **Profiling Issues**: Update ydata-profiling
   ```bash
   pip install --upgrade ydata-profiling
   ```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the generated reports for specific error details
3. Ensure your data format is supported (CSV, Excel, JSON, Parquet)

---

**Happy Data Analyzing! 🎉** 