#!/usr/bin/env python3
"""
Main Analytics Script for Comprehensive Data Testing and Analysis
Combines analytics model and data testing utilities
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Import our custom modules
from analytics_model import DataAnalyticsModel
from data_testing_utils import DataTestingUtils

class ComprehensiveDataAnalytics:
    """
    Main class that combines analytics and testing capabilities
    """
    
    def __init__(self):
        self.analytics_model = None
        self.testing_utils = DataTestingUtils()
        self.results = {}
    
    def analyze_data(self, 
                    data_source: str, 
                    output_dir: str = "analytics_output",
                    generate_plots: bool = True,
                    generate_dashboard: bool = True,
                    generate_profiling: bool = True) -> Dict[str, Any]:
        """
        Comprehensive data analysis pipeline
        
        Args:
            data_source: Path to data file
            output_dir: Directory to save results
            generate_plots: Whether to generate static plots
            generate_dashboard: Whether to generate interactive dashboard
            generate_profiling: Whether to generate profiling report
        """
        print("🚀 Starting comprehensive data analysis...")
        
        # Initialize analytics model
        self.analytics_model = DataAnalyticsModel(data_source)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run comprehensive analysis
        analysis_results = self.analytics_model.run_comprehensive_analysis(output_dir)
        
        # Run data testing
        print("🧪 Running data quality tests...")
        test_results = self.testing_utils.generate_test_report(
            self.analytics_model.data, 
            output_dir
        )
        
        # Combine results
        self.results = {
            'analysis': analysis_results,
            'testing': test_results,
            'summary': self._generate_summary(analysis_results, test_results)
        }
        
        # Save combined results
        self._save_results(output_dir)
        
        print(f"✅ Analysis complete! Results saved to: {output_dir}")
        return self.results
    
    def _generate_summary(self, analysis_results: Dict, test_results: Dict) -> Dict[str, Any]:
        """
        Generate a summary of all results
        """
        data_quality = analysis_results.get('quality_report', {})
        testing_results = test_results.get('summary', {})
        
        summary = {
            'data_overview': {
                'rows': data_quality.get('basic_info', {}).get('rows', 0),
                'columns': data_quality.get('basic_info', {}).get('columns', 0),
                'missing_percentage': data_quality.get('missing_data', {}).get('missing_percentage', 0)
            },
            'quality_score': self._calculate_quality_score(data_quality, testing_results),
            'issues_summary': {
                'critical_issues': testing_results.get('critical_issues', 0),
                'warnings': testing_results.get('warnings', 0),
                'total_issues': testing_results.get('total_issues', 0)
            },
            'recommendations': testing_results.get('recommendations', [])
        }
        
        return summary
    
    def _calculate_quality_score(self, data_quality: Dict, testing_results: Dict) -> float:
        """
        Calculate an overall data quality score (0-100)
        """
        score = 100.0
        
        # Deduct points for various issues
        missing_percentage = data_quality.get('missing_data', {}).get('missing_percentage', 0)
        score -= missing_percentage * 0.5  # Each 1% missing data reduces score by 0.5
        
        duplicate_percentage = (data_quality.get('basic_info', {}).get('duplicate_rows', 0) / 
                              data_quality.get('basic_info', {}).get('rows', 1)) * 100
        score -= duplicate_percentage * 0.3  # Each 1% duplicates reduces score by 0.3
        
        # Deduct for critical issues and warnings
        score -= testing_results.get('critical_issues', 0) * 10  # Each critical issue reduces score by 10
        score -= testing_results.get('warnings', 0) * 2  # Each warning reduces score by 2
        
        return max(0, score)
    
    def _save_results(self, output_dir: str) -> None:
        """
        Save all results to files
        """
        import json
        
        # Save combined results
        with open(f"{output_dir}/combined_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        with open(f"{output_dir}/analysis_summary.txt", 'w') as f:
            f.write("=== COMPREHENSIVE DATA ANALYSIS SUMMARY ===\n\n")
            
            summary = self.results['summary']
            f.write(f"Data Overview:\n")
            f.write(f"  - Rows: {summary['data_overview']['rows']}\n")
            f.write(f"  - Columns: {summary['data_overview']['columns']}\n")
            f.write(f"  - Missing Data: {summary['data_overview']['missing_percentage']:.2f}%\n\n")
            
            f.write(f"Quality Score: {summary['quality_score']:.1f}/100\n\n")
            
            f.write(f"Issues Found:\n")
            f.write(f"  - Critical Issues: {summary['issues_summary']['critical_issues']}\n")
            f.write(f"  - Warnings: {summary['issues_summary']['warnings']}\n")
            f.write(f"  - Total Issues: {summary['issues_summary']['total_issues']}\n\n")
            
            if summary['recommendations']:
                f.write("Recommendations:\n")
                for rec in summary['recommendations']:
                    f.write(f"  - {rec}\n")
    
    def quick_analysis(self, data_source: str) -> Dict[str, Any]:
        """
        Quick analysis without generating plots
        """
        print("⚡ Running quick analysis...")
        
        # Initialize model
        self.analytics_model = DataAnalyticsModel(data_source)
        
        # Get basic info
        data_summary = self.analytics_model.get_data_summary()
        quality_report = self.analytics_model.analyze_data_quality()
        
        # Run basic tests
        test_results = self.testing_utils.generate_test_report(self.analytics_model.data)
        
        return {
            'data_summary': data_summary,
            'quality_report': quality_report,
            'test_results': test_results
        }

def create_sample_data(output_path: str = "sample_data.csv") -> str:
    """
    Create sample data for testing
    """
    print("📊 Creating sample data...")
    
    # Create realistic sample data with various data types and issues
    np.random.seed(42)
    
    data = pd.DataFrame({
        'customer_id': range(1000),
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.lognormal(10, 0.5, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'purchase_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'product_name': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 1000),
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000),
        'is_premium': np.random.choice([True, False], 1000),
        'notes': [f'Note for customer {i}' for i in range(1000)]
    })
    
    # Add some data quality issues
    data.loc[0:50, 'age'] = np.nan  # Missing values
    data.loc[100:150, 'income'] = -1000  # Negative income
    data.loc[200:250, 'age'] = 200  # Impossible age
    data.loc[300:350, 'category'] = 'INVALID'  # Invalid category
    data.loc[400:450, 'rating'] = 10  # Invalid rating
    data.loc[500:550, 'purchase_date'] = pd.NaT  # Missing dates
    
    # Add some duplicates
    data = pd.concat([data, data.iloc[0:10]], ignore_index=True)
    
    # Save to file
    data.to_csv(output_path, index=False)
    print(f"✅ Sample data created: {output_path}")
    
    return output_path

def main():
    """
    Main function to run the analytics pipeline
    """
    parser = argparse.ArgumentParser(description="Comprehensive Data Analytics and Testing")
    parser.add_argument("--data", "-d", help="Path to data file")
    parser.add_argument("--output", "-o", default="analytics_output", help="Output directory")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick analysis only")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data for testing")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip generating dashboard")
    parser.add_argument("--no-profiling", action="store_true", help="Skip generating profiling report")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        data_path = create_sample_data()
        print(f"Sample data created at: {data_path}")
        return
    
    # Check if data file is provided
    if not args.data:
        print("❌ Error: Please provide a data file using --data or --create-sample to generate sample data")
        sys.exit(1)
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Initialize analytics system
    analytics = ComprehensiveDataAnalytics()
    
    try:
        if args.quick:
            # Quick analysis
            results = analytics.quick_analysis(args.data)
            print("\n=== QUICK ANALYSIS RESULTS ===")
            print(f"Data shape: {results['data_summary']['shape']}")
            print(f"Missing data: {results['quality_report']['missing_data']['missing_percentage']:.2f}%")
            print(f"Quality issues found: {results['test_results']['summary']['total_issues']}")
        else:
            # Full analysis
            results = analytics.analyze_data(
                data_source=args.data,
                output_dir=args.output,
                generate_plots=not args.no_plots,
                generate_dashboard=not args.no_dashboard,
                generate_profiling=not args.no_profiling
            )
            
            # Print summary
            summary = results['summary']
            print("\n=== ANALYSIS SUMMARY ===")
            print(f"Quality Score: {summary['quality_score']:.1f}/100")
            print(f"Critical Issues: {summary['issues_summary']['critical_issues']}")
            print(f"Warnings: {summary['issues_summary']['warnings']}")
            
            if summary['recommendations']:
                print("\nRecommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
    
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 