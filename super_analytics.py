#!/usr/bin/env python3
"""
Super Analytics System
Comprehensive data analytics with cleaning, testing, advanced analytics, and dashboards
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

# Import all our modules
from analytics_model import DataAnalyticsModel
from data_testing_utils import DataTestingUtils
from advanced_analytics import AdvancedAnalytics
from data_cleaner import DataCleaner
from dashboard_generator import DashboardGenerator

class SuperAnalytics:
    """
    Super comprehensive analytics system that combines all capabilities
    """
    
    def __init__(self):
        self.analytics_model = None
        self.testing_utils = DataTestingUtils()
        self.advanced_analytics = AdvancedAnalytics()
        self.data_cleaner = DataCleaner()
        self.dashboard_generator = DashboardGenerator()
        self.results = {}
    
    def run_comprehensive_analysis(self, data_source: str, 
                                 output_dir: str = "super_analytics_output",
                                 clean_data: bool = True,
                                 run_advanced: bool = True,
                                 generate_dashboards: bool = True) -> Dict[str, Any]:
        """
        Run the complete super analytics pipeline
        """
        print("🚀 Starting Super Analytics Pipeline...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load and analyze data
        print("📊 Step 1: Loading and analyzing data...")
        self.analytics_model = DataAnalyticsModel(data_source)
        basic_analysis = self.analytics_model.run_comprehensive_analysis(output_dir)
        
        # Step 2: Data testing
        print("🧪 Step 2: Running data quality tests...")
        test_results = self.testing_utils.generate_test_report(self.analytics_model.data, output_dir)
        
        # Step 3: Data cleaning (optional)
        cleaned_data = self.analytics_model.data
        cleaning_stats = None
        if clean_data:
            print("🧹 Step 3: Cleaning data...")
            cleaned_data, cleaning_stats = self.data_cleaner.comprehensive_clean(self.analytics_model.data)
            
            # Save cleaned data
            cleaned_data.to_csv(f"{output_dir}/cleaned_data.csv", index=False)
        
        # Step 4: Advanced analytics (optional)
        advanced_results = None
        if run_advanced:
            print("🚀 Step 4: Running advanced analytics...")
            try:
                advanced_results = self.advanced_analytics.run_comprehensive_advanced_analysis(
                    cleaned_data, output_dir
                )
            except Exception as e:
                print(f"⚠️ Advanced analytics failed: {str(e)}")
                advanced_results = {"error": str(e)}
        
        # Step 5: Generate dashboards (optional)
        dashboard_results = None
        if generate_dashboards:
            print("🎨 Step 5: Generating dashboards...")
            try:
                dashboard_results = self.dashboard_generator.generate_all_dashboards(
                    cleaned_data, basic_analysis, advanced_results, 
                    quality_results=test_results, save_path=output_dir
                )
            except Exception as e:
                print(f"⚠️ Dashboard generation failed: {str(e)}")
                dashboard_results = {"error": str(e)}
        
        # Combine all results
        self.results = {
            'basic_analysis': basic_analysis,
            'test_results': test_results,
            'cleaning_stats': cleaning_stats,
            'advanced_results': advanced_results,
            'dashboard_results': dashboard_results,
            'summary': self._generate_super_summary(basic_analysis, test_results, cleaning_stats, advanced_results)
        }
        
        # Save combined results
        self._save_super_results(output_dir)
        
        print(f"✅ Super Analytics complete! Results saved to: {output_dir}")
        return self.results
    
    def _generate_super_summary(self, basic_analysis: Dict, test_results: Dict, 
                              cleaning_stats: Dict, advanced_results: Dict) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all results
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_overview': {
                'original_rows': basic_analysis.get('quality_report', {}).get('basic_info', {}).get('rows', 0),
                'original_columns': basic_analysis.get('quality_report', {}).get('basic_info', {}).get('columns', 0),
                'cleaned_rows': cleaning_stats.get('cleaned_shape', [0, 0])[0] if cleaning_stats else 0,
                'cleaned_columns': cleaning_stats.get('cleaned_shape', [0, 0])[1] if cleaning_stats else 0,
                'missing_percentage': basic_analysis.get('quality_report', {}).get('missing_data', {}).get('missing_percentage', 0)
            },
            'quality_metrics': {
                'original_quality_score': basic_analysis.get('summary', {}).get('quality_score', 0),
                'cleaned_quality_score': cleaning_stats.get('quality_report', {}).get('quality_score', 0) if cleaning_stats else 0,
                'critical_issues': test_results.get('summary', {}).get('critical_issues', 0),
                'warnings': test_results.get('summary', {}).get('warnings', 0),
                'total_issues': test_results.get('summary', {}).get('total_issues', 0)
            },
            'advanced_analytics': {
                'anomalies_detected': advanced_results.get('anomaly_detection', {}).get('anomaly_count', 0) if advanced_results and 'error' not in advanced_results.get('anomaly_detection', {}) else 0,
                'clusters_found': advanced_results.get('clustering', {}).get('n_clusters', 0) if advanced_results and 'error' not in advanced_results.get('clustering', {}) else 0,
                'variance_explained': advanced_results.get('pca_analysis', {}).get('total_variance_explained', 0) if advanced_results and 'error' not in advanced_results.get('pca_analysis', {}) else 0
            },
            'recommendations': self._generate_super_recommendations(basic_analysis, test_results, cleaning_stats, advanced_results)
        }
        
        return summary
    
    def _generate_super_recommendations(self, basic_analysis: Dict, test_results: Dict,
                                      cleaning_stats: Dict, advanced_results: Dict) -> List[str]:
        """
        Generate comprehensive recommendations
        """
        recommendations = []
        
        # Basic quality recommendations
        quality_score = basic_analysis.get('summary', {}).get('quality_score', 0)
        if quality_score < 50:
            recommendations.append("CRITICAL: Data quality is very poor. Manual review required.")
        elif quality_score < 70:
            recommendations.append("WARNING: Data quality needs improvement. Consider data cleaning.")
        
        # Cleaning recommendations
        if cleaning_stats:
            rows_removed = cleaning_stats.get('rows_removed', 0)
            if rows_removed > 0:
                recommendations.append(f"INFO: {rows_removed} rows were cleaned/removed during processing.")
        
        # Advanced analytics recommendations
        if advanced_results and 'error' not in advanced_results:
            anomaly_count = advanced_results.get('anomaly_detection', {}).get('anomaly_count', 0)
            if anomaly_count > 0:
                recommendations.append(f"INFO: {anomaly_count} anomalies detected in the data.")
            
            cluster_count = advanced_results.get('clustering', {}).get('n_clusters', 0)
            if cluster_count > 0:
                recommendations.append(f"INFO: {cluster_count} natural clusters found in the data.")
        
        # Test results recommendations
        test_recommendations = test_results.get('summary', {}).get('recommendations', [])
        recommendations.extend(test_recommendations)
        
        return recommendations
    
    def _save_super_results(self, output_dir: str) -> None:
        """
        Save all super analytics results
        """
        # Save combined results
        with open(f"{output_dir}/super_analytics_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        with open(f"{output_dir}/super_analytics_summary.txt", 'w') as f:
            f.write("=== SUPER ANALYTICS SUMMARY ===\n\n")
            
            summary = self.results['summary']
            
            f.write("Data Overview:\n")
            f.write(f"  - Original Shape: {summary['data_overview']['original_rows']} rows, {summary['data_overview']['original_columns']} columns\n")
            f.write(f"  - Cleaned Shape: {summary['data_overview']['cleaned_rows']} rows, {summary['data_overview']['cleaned_columns']} columns\n")
            f.write(f"  - Missing Data: {summary['data_overview']['missing_percentage']:.2f}%\n\n")
            
            f.write("Quality Metrics:\n")
            f.write(f"  - Original Quality Score: {summary['quality_metrics']['original_quality_score']:.1f}/100\n")
            f.write(f"  - Cleaned Quality Score: {summary['quality_metrics']['cleaned_quality_score']:.1f}/100\n")
            f.write(f"  - Critical Issues: {summary['quality_metrics']['critical_issues']}\n")
            f.write(f"  - Warnings: {summary['quality_metrics']['warnings']}\n\n")
            
            f.write("Advanced Analytics:\n")
            f.write(f"  - Anomalies Detected: {summary['advanced_analytics']['anomalies_detected']}\n")
            f.write(f"  - Clusters Found: {summary['advanced_analytics']['clusters_found']}\n")
            f.write(f"  - Variance Explained: {summary['advanced_analytics']['variance_explained']:.1f}%\n\n")
            
            if summary['recommendations']:
                f.write("Recommendations:\n")
                for rec in summary['recommendations']:
                    f.write(f"  - {rec}\n")
    
    def quick_analysis(self, data_source: str) -> Dict[str, Any]:
        """
        Quick analysis without advanced features
        """
        print("⚡ Running quick super analysis...")
        
        # Load data
        self.analytics_model = DataAnalyticsModel(data_source)
        
        # Basic analysis
        basic_analysis = self.analytics_model.analyze_data_quality()
        
        # Quick testing
        test_results = self.testing_utils.generate_test_report(self.analytics_model.data)
        
        # Quick cleaning
        cleaned_data, cleaning_stats = self.data_cleaner.comprehensive_clean(self.analytics_model.data)
        
        return {
            'data_summary': self.analytics_model.get_data_summary(),
            'quality_report': basic_analysis,
            'test_results': test_results,
            'cleaning_stats': cleaning_stats
        }

def create_super_sample_data(output_path: str = "super_sample_data.csv") -> str:
    """
    Create comprehensive sample data for testing
    """
    print("📊 Creating super sample data...")
    
    np.random.seed(42)
    
    # Create time series data with trends and anomalies
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    trend = np.linspace(0, 100, 1000)
    noise = np.random.normal(0, 10, 1000)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(1000) / 365)
    
    # Add anomalies
    anomalies = np.zeros(1000)
    anomaly_indices = [100, 300, 500, 700, 900]
    for idx in anomaly_indices:
        anomalies[idx] = np.random.normal(50, 20)
    
    values = trend + noise + seasonal + anomalies
    
    # Create comprehensive DataFrame
    data = pd.DataFrame({
        'customer_id': range(1000),
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.lognormal(10, 0.5, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'purchase_date': dates,
        'purchase_value': values,
        'product_name': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 1000),
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000),
        'is_premium': np.random.choice([True, False], 1000),
        'notes': [f'Note for customer {i}' for i in range(1000)],
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.normal(0, 1, 1000)
    })
    
    # Add data quality issues
    data.loc[0:50, 'age'] = np.nan  # Missing values
    data.loc[100:150, 'income'] = -1000  # Negative income
    data.loc[200:250, 'age'] = 200  # Impossible age
    data.loc[300:350, 'category'] = 'INVALID'  # Invalid category
    data.loc[400:450, 'rating'] = 10  # Invalid rating
    data.loc[500:550, 'purchase_date'] = pd.NaT  # Missing dates
    
    # Add duplicates
    data = pd.concat([data, data.iloc[0:10]], ignore_index=True)
    
    # Add text issues
    data.loc[600:650, 'notes'] = '  MULTIPLE   SPACES  '
    data.loc[700:750, 'notes'] = 'UPPERCASE TEXT'
    
    # Save to file
    data.to_csv(output_path, index=False)
    print(f"✅ Super sample data created: {output_path}")
    
    return output_path

def main():
    """
    Main function for super analytics
    """
    parser = argparse.ArgumentParser(description="Super Analytics System")
    parser.add_argument("--data", "-d", help="Path to data file")
    parser.add_argument("--output", "-o", default="super_analytics_output", help="Output directory")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick analysis only")
    parser.add_argument("--create-sample", action="store_true", help="Create super sample data")
    parser.add_argument("--no-clean", action="store_true", help="Skip data cleaning")
    parser.add_argument("--no-advanced", action="store_true", help="Skip advanced analytics")
    parser.add_argument("--no-dashboards", action="store_true", help="Skip dashboard generation")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        data_path = create_super_sample_data()
        print(f"Super sample data created at: {data_path}")
        return
    
    # Check if data file is provided
    if not args.data:
        print("❌ Error: Please provide a data file using --data or --create-sample to generate sample data")
        sys.exit(1)
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Initialize super analytics
    super_analytics = SuperAnalytics()
    
    try:
        if args.quick:
            # Quick analysis
            results = super_analytics.quick_analysis(args.data)
            print("\n=== QUICK SUPER ANALYSIS RESULTS ===")
            print(f"Data shape: {results['data_summary']['shape']}")
            print(f"Missing data: {results['quality_report']['missing_data']['missing_percentage']:.2f}%")
            print(f"Quality issues: {results['test_results']['summary']['total_issues']}")
            print(f"Cleaned quality score: {results['cleaning_stats']['quality_report']['quality_score']:.1f}/100")
        else:
            # Full super analysis
            results = super_analytics.run_comprehensive_analysis(
                data_source=args.data,
                output_dir=args.output,
                clean_data=not args.no_clean,
                run_advanced=not args.no_advanced,
                generate_dashboards=not args.no_dashboards
            )
            
            # Print summary
            summary = results['summary']
            print("\n=== SUPER ANALYTICS SUMMARY ===")
            print(f"Original Quality Score: {summary['quality_metrics']['original_quality_score']:.1f}/100")
            print(f"Cleaned Quality Score: {summary['quality_metrics']['cleaned_quality_score']:.1f}/100")
            print(f"Critical Issues: {summary['quality_metrics']['critical_issues']}")
            print(f"Warnings: {summary['quality_metrics']['warnings']}")
            print(f"Anomalies Detected: {summary['advanced_analytics']['anomalies_detected']}")
            print(f"Clusters Found: {summary['advanced_analytics']['clusters_found']}")
            
            if summary['recommendations']:
                print("\nRecommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
    
    except Exception as e:
        print(f"❌ Error during super analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 