#!/usr/bin/env python3
"""
Advanced Analytics Module with ML Capabilities
Extends the basic analytics with machine learning features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """
    Advanced analytics with machine learning capabilities
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scalers = {}
    
    def detect_anomalies(self, data: pd.DataFrame, columns: List[str] = None, 
                        contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data
        numeric_data = data[columns].dropna()
        
        if len(numeric_data) == 0:
            return {"error": "No numeric data available for anomaly detection"}
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Train isolation forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        
        # Get anomaly scores
        anomaly_scores = iso_forest.decision_function(scaled_data)
        
        # Create results
        results = {
            'anomaly_indices': np.where(anomaly_labels == -1)[0].tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_count': len(np.where(anomaly_labels == -1)[0]),
            'total_samples': len(numeric_data),
            'anomaly_percentage': (len(np.where(anomaly_labels == -1)[0]) / len(numeric_data)) * 100,
            'columns_used': columns
        }
        
        # Store model
        self.models['anomaly_detector'] = iso_forest
        self.scalers['anomaly_scaler'] = scaler
        
        return results
    
    def perform_clustering(self, data: pd.DataFrame, columns: List[str] = None,
                         n_clusters: int = 3, method: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform clustering analysis
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data
        numeric_data = data[columns].dropna()
        
        if len(numeric_data) == 0:
            return {"error": "No numeric data available for clustering"}
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Perform clustering
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            cluster_centers = kmeans.cluster_centers_
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            
            results = {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': cluster_centers.tolist(),
                'silhouette_score': silhouette_avg,
                'n_clusters': n_clusters,
                'method': method,
                'columns_used': columns
            }
            
            self.models['cluster_model'] = kmeans
            self.scalers['cluster_scaler'] = scaler
        
        return results
    
    def perform_pca_analysis(self, data: pd.DataFrame, columns: List[str] = None,
                           n_components: int = 2) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data
        numeric_data = data[columns].dropna()
        
        if len(numeric_data) == 0:
            return {"error": "No numeric data available for PCA"}
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, len(columns)))
        pca_result = pca.fit_transform(scaled_data)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        results = {
            'pca_result': pca_result.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'n_components': len(pca_result[0]),
            'columns_used': columns,
            'total_variance_explained': cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
        }
        
        self.models['pca_model'] = pca
        self.scalers['pca_scaler'] = scaler
        
        return results
    
    def predict_missing_values(self, data: pd.DataFrame, target_column: str,
                             method: str = 'random_forest') -> Dict[str, Any]:
        """
        Predict missing values using machine learning
        """
        if target_column not in data.columns:
            return {"error": f"Target column '{target_column}' not found"}
        
        # Prepare features (all numeric columns except target)
        feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        
        if len(feature_columns) == 0:
            return {"error": "No feature columns available for prediction"}
        
        # Split data into known and unknown values
        known_data = data.dropna(subset=[target_column])
        unknown_data = data[data[target_column].isna()]
        
        if len(known_data) == 0:
            return {"error": "No known values for training"}
        
        if len(unknown_data) == 0:
            return {"error": "No missing values to predict"}
        
        # Prepare training data
        X_train = known_data[feature_columns].fillna(0)
        y_train = known_data[target_column]
        
        # Train model
        if method == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict missing values
            X_predict = unknown_data[feature_columns].fillna(0)
            predictions = model.predict(X_predict)
            
            results = {
                'predictions': predictions.tolist(),
                'indices_with_predictions': unknown_data.index.tolist(),
                'method': method,
                'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
                'training_samples': len(known_data),
                'prediction_samples': len(unknown_data)
            }
            
            self.models['missing_value_predictor'] = model
        
        return results
    
    def generate_correlation_network(self, data: pd.DataFrame, 
                                   threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate correlation network analysis
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) >= 0.7 else 'moderate'
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'threshold': threshold,
            'total_strong_correlations': len(strong_correlations)
        }
    
    def analyze_temporal_patterns(self, data: pd.DataFrame, 
                                date_column: str, value_column: str) -> Dict[str, Any]:
        """
        Analyze temporal patterns in time series data
        """
        if date_column not in data.columns or value_column not in data.columns:
            return {"error": "Date or value column not found"}
        
        # Convert to datetime
        try:
            data[date_column] = pd.to_datetime(data[date_column])
        except:
            return {"error": "Could not convert date column to datetime"}
        
        # Sort by date
        temporal_data = data[[date_column, value_column]].dropna().sort_values(date_column)
        
        if len(temporal_data) == 0:
            return {"error": "No valid temporal data"}
        
        # Calculate temporal statistics
        temporal_stats = {
            'start_date': temporal_data[date_column].min(),
            'end_date': temporal_data[date_column].max(),
            'date_range_days': (temporal_data[date_column].max() - temporal_data[date_column].min()).days,
            'total_observations': len(temporal_data),
            'mean_value': temporal_data[value_column].mean(),
            'std_value': temporal_data[value_column].std(),
            'min_value': temporal_data[value_column].min(),
            'max_value': temporal_data[value_column].max()
        }
        
        # Calculate trends (simple linear trend)
        temporal_data['days_since_start'] = (temporal_data[date_column] - temporal_data[date_column].min()).dt.days
        
        if len(temporal_data) > 1:
            # Simple linear trend
            x = temporal_data['days_since_start'].values.reshape(-1, 1)
            y = temporal_data[value_column].values
            
            from sklearn.linear_model import LinearRegression
            trend_model = LinearRegression()
            trend_model.fit(x, y)
            
            trend_slope = trend_model.coef_[0]
            trend_intercept = trend_model.intercept_
            trend_r2 = trend_model.score(x, y)
            
            temporal_stats.update({
                'trend_slope': trend_slope,
                'trend_intercept': trend_intercept,
                'trend_r2': trend_r2,
                'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'trend_strength': 'strong' if abs(trend_r2) > 0.7 else 'weak'
            })
        
        return {
            'temporal_statistics': temporal_stats,
            'data_points': temporal_data[[date_column, value_column]].to_dict('records')
        }
    
    def generate_advanced_visualizations(self, data: pd.DataFrame, 
                                       save_path: str = None) -> Dict[str, Any]:
        """
        Generate advanced visualizations
        """
        viz_results = {}
        
        # 1. Anomaly detection visualization
        anomaly_results = self.detect_anomalies(data)
        if 'error' not in anomaly_results:
            plt.figure(figsize=(12, 6))
            anomaly_scores = anomaly_results['anomaly_scores']
            plt.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(np.percentile(anomaly_scores, 90), color='red', linestyle='--', 
                       label='90th percentile threshold')
            plt.title('Anomaly Score Distribution')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/anomaly_distribution.png", dpi=300, bbox_inches='tight')
            viz_results['anomaly_distribution'] = plt.gcf()
            plt.close()
        
        # 2. Clustering visualization
        cluster_results = self.perform_clustering(data, n_clusters=3)
        if 'error' not in cluster_results:
            # Use PCA for 2D visualization
            pca_results = self.perform_pca_analysis(data, n_components=2)
            if 'error' not in pca_results:
                pca_data = np.array(pca_results['pca_result'])
                cluster_labels = cluster_results['cluster_labels']
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                                   c=cluster_labels, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter)
                plt.title('Clustering Results (PCA 2D projection)')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.tight_layout()
                if save_path:
                    plt.savefig(f"{save_path}/clustering_visualization.png", dpi=300, bbox_inches='tight')
                viz_results['clustering'] = plt.gcf()
                plt.close()
        
        # 3. Correlation network heatmap
        corr_network = self.generate_correlation_network(data)
        if 'error' not in corr_network:
            correlation_matrix = pd.DataFrame(corr_network['correlation_matrix'])
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Advanced Correlation Matrix')
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/advanced_correlation_matrix.png", dpi=300, bbox_inches='tight')
            viz_results['advanced_correlation'] = plt.gcf()
            plt.close()
        
        return viz_results
    
    def run_comprehensive_advanced_analysis(self, data: pd.DataFrame, 
                                          save_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive advanced analysis
        """
        print("🚀 Starting advanced analytics...")
        
        results = {
            'anomaly_detection': self.detect_anomalies(data),
            'clustering': self.perform_clustering(data),
            'pca_analysis': self.perform_pca_analysis(data),
            'correlation_network': self.generate_correlation_network(data),
            'advanced_visualizations': self.generate_advanced_visualizations(data, save_path)
        }
        
        # Try temporal analysis if date columns exist
        date_columns = data.select_dtypes(include=['datetime64']).columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(date_columns) > 0 and len(numeric_columns) > 0:
            for date_col in date_columns[:1]:  # Use first date column
                for num_col in numeric_columns[:1]:  # Use first numeric column
                    temporal_results = self.analyze_temporal_patterns(data, date_col, num_col)
                    if 'error' not in temporal_results:
                        results['temporal_analysis'] = temporal_results
                        break
                break
        
        # Try missing value prediction for numeric columns with missing values
        numeric_cols_with_missing = data.select_dtypes(include=[np.number]).columns[
            data.select_dtypes(include=[np.number]).isnull().any()
        ]
        
        if len(numeric_cols_with_missing) > 0:
            target_col = numeric_cols_with_missing[0]
            prediction_results = self.predict_missing_values(data, target_col)
            if 'error' not in prediction_results:
                results['missing_value_prediction'] = prediction_results
        
        print("✅ Advanced analytics complete!")
        return results

# Example usage
def test_advanced_analytics():
    """
    Test the advanced analytics module
    """
    # Create sample data with more complex patterns
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # Create data with trends and anomalies
    trend = np.linspace(0, 100, 1000)
    noise = np.random.normal(0, 10, 1000)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(1000) / 365)
    
    # Add some anomalies
    anomalies = np.zeros(1000)
    anomaly_indices = [100, 300, 500, 700, 900]
    for idx in anomaly_indices:
        anomalies[idx] = np.random.normal(50, 20)
    
    values = trend + noise + seasonal + anomalies
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.normal(0, 1, 1000)
    })
    
    # Add some missing values
    data.loc[0:50, 'value'] = np.nan
    
    # Test advanced analytics
    advanced = AdvancedAnalytics()
    results = advanced.run_comprehensive_advanced_analysis(data, "advanced_output")
    
    print("🎉 Advanced analytics test completed!")
    return advanced, results

if __name__ == "__main__":
    advanced, results = test_advanced_analytics() 