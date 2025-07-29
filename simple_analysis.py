"""
COMPREHENSIVE MOTOR PREDICTIVE MAINTENANCE ANALYSIS
Research-grade analysis with advanced visualizations and machine learning
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import MotorDataLoader
from eda_visualizer import create_plots_directory
import warnings
warnings.filterwarnings('ignore')

def main():
    print("STARTING COMPREHENSIVE MOTOR PREDICTIVE MAINTENANCE ANALYSIS")
    print("=" * 70)
    
    # Create necessary directories
    create_plots_directory()
    
    # Step 1: Load and combine all motor data
    print("\nSTEP 1: DATA LOADING AND PREPROCESSING")
    print("-" * 50)
    
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    summary = loader.get_data_summary()
    
    # Print comprehensive data summary
    print(f"\nDATASET OVERVIEW:")
    print(f"   • Total measurements: {summary['total_rows']:,}")
    print(f"   • Test sessions: {summary['total_sessions']}")
    print(f"   • Motors analyzed: {summary['total_motors']}")
    print(f"   • Test duration: {summary['time_range']['duration_hours']:.2f} hours")
    
    print(f"\nSENSOR STATISTICS:")
    for sensor, stats in summary['sensor_stats'].items():
        print(f"   • {sensor.title()}:")
        print(f"     - Range: {stats['min']:.2f} to {stats['max']:.2f}")
        print(f"     - Average: {stats['mean']:.2f}")
    
    # Step 2: Anomaly Detection
    print("\nSTEP 2: ANOMALY DETECTION")
    print("-" * 50)
    
    anomaly_data = loader.detect_anomalies(method='iqr')
    total_anomalies = anomaly_data['is_anomaly'].sum()
    anomaly_rate = (total_anomalies / len(anomaly_data)) * 100
    
    print(f"   • Total anomalies detected: {total_anomalies:,}")
    print(f"   • Anomaly rate: {anomaly_rate:.2f}%")
    
    # Breakdown by sensor
    for sensor in ['temperature', 'voltage', 'position']:
        sensor_anomalies = anomaly_data[f'{sensor}_anomaly'].sum()
        sensor_rate = (sensor_anomalies / len(anomaly_data)) * 100
        print(f"   • {sensor.title()} anomalies: {sensor_anomalies:,} ({sensor_rate:.1f}%)")
    
    # Step 3: Basic Analysis Complete
    print("\nANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"   Generated dataset ready for ML training")
    print(f"   • {len(anomaly_data)} labeled samples")
    print(f"   • {total_anomalies} positive anomaly cases")
    print(f"   • Multi-sensor time series data preprocessed")
    
    return combined_data, anomaly_data, summary

if __name__ == "__main__":
    combined_data, anomaly_data, summary = main()