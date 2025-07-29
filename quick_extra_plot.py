"""
Quick extra plot - Feature importance with cool styling
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_loader import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

# Cool dark styling
plt.style.use('dark_background')

def main():
    print("Creating feature importance radar chart...")
    
    # Load data quickly
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    anomaly_data = loader.detect_anomalies(method='iqr')
    
    # Quick feature prep
    feature_cols = ['temperature', 'voltage', 'position', 'relative_time']
    anomaly_data['temp_rolling_mean'] = anomaly_data.groupby(['session', 'motor_id'])['temperature'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean())
    anomaly_data['voltage_rolling_std'] = anomaly_data.groupby(['session', 'motor_id'])['voltage'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std())
    
    le_session = LabelEncoder()
    le_motor = LabelEncoder()
    anomaly_data['session_encoded'] = le_session.fit_transform(anomaly_data['session'])
    anomaly_data['motor_encoded'] = le_motor.fit_transform(anomaly_data['motor_id'])
    
    feature_cols.extend(['temp_rolling_mean', 'voltage_rolling_std', 'session_encoded', 'motor_encoded'])
    
    X = anomaly_data[feature_cols].fillna(0)
    y = anomaly_data['is_anomaly'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Create enhanced feature importance plot
    importances = rf.feature_importances_
    feature_names = X.columns
    
    plt.figure(figsize=(14, 8), facecolor='black')
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Create gradient colors
    colors = plt.cm.plasma(np.linspace(0, 1, len(importances)))
    
    # Create bar plot
    bars = plt.bar(range(len(importances)), importances[indices], color=colors[indices])
    
    # Add glow effect
    for bar, importance in zip(bars, importances[indices]):
        height = bar.get_height()
        # Add value label
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{importance:.3f}', ha='center', va='bottom', 
                color='white', fontweight='bold', fontsize=11)
        
        # Add glow effect for top features
        if importance > 0.1:
            bar.set_edgecolor('white')
            bar.set_linewidth(2)
    
    plt.title('Feature Importance Analysis - Motor Anomaly Detection', 
              color='white', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Features', color='white', fontsize=14)
    plt.ylabel('Importance Score', color='white', fontsize=14)
    
    # Set feature names as x-tick labels
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
               rotation=45, ha='right', color='white', fontsize=10)
    plt.yticks(color='white')
    
    # Add grid
    plt.grid(True, alpha=0.3, color='white')
    
    # Style the plot
    plt.gca().set_facecolor('black')
    plt.tight_layout()
    
    plt.savefig('plots/enhanced_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print("Enhanced feature importance plot created!")
    print("File saved: plots/enhanced_feature_importance.png")
    
    return rf, importances

if __name__ == "__main__":
    main()