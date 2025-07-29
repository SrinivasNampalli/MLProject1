"""
Quick ML Demo for Motor Predictive Maintenance
Faster training with reduced hyperparameter tuning for demonstration
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from data_loader import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

def main():
    print("QUICK MACHINE LEARNING DEMO")
    print("=" * 50)
    
    # Load data
    print("\nLoading motor sensor data...")
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    anomaly_data = loader.detect_anomalies(method='iqr')
    
    print(f"   Loaded {len(anomaly_data)} samples")
    print(f"   Anomaly rate: {(anomaly_data['is_anomaly'].sum() / len(anomaly_data) * 100):.2f}%")
    
    # Prepare features (simplified)
    print("\nFeature Engineering...")
    feature_cols = ['temperature', 'voltage', 'position', 'relative_time']
    
    # Add rolling features
    anomaly_data['temp_rolling_mean'] = anomaly_data.groupby(['session', 'motor_id'])['temperature'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    anomaly_data['voltage_rolling_std'] = anomaly_data.groupby(['session', 'motor_id'])['voltage'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    
    # Encode categorical variables
    le_session = LabelEncoder()
    le_motor = LabelEncoder()
    
    anomaly_data['session_encoded'] = le_session.fit_transform(anomaly_data['session'])
    anomaly_data['motor_encoded'] = le_motor.fit_transform(anomaly_data['motor_id'])
    
    feature_cols.extend(['temp_rolling_mean', 'voltage_rolling_std', 'session_encoded', 'motor_encoded'])
    
    print(f"   Created {len(feature_cols)} features")
    
    # Prepare data
    X = anomaly_data[feature_cols].fillna(0)
    y = anomaly_data['is_anomaly'].astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train Random Forest (simplified)
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    
    rf_pred = rf.predict(X_test_scaled)
    rf_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    print(f"   Random Forest AUC: {rf_auc:.4f}")
    
    # Train XGBoost (simplified)
    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc'
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    
    print(f"   XGBoost AUC: {xgb_auc:.4f}")
    
    # Model comparison
    print("\nMODEL COMPARISON")
    print("-" * 30)
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC:       {xgb_auc:.4f}")
    
    best_model = "Random Forest" if rf_auc > xgb_auc else "XGBoost"
    best_auc = max(rf_auc, xgb_auc)
    
    print(f"\nBest Model: {best_model} (AUC: {best_auc:.4f})")
    
    # Feature importance
    print("\nTOP FEATURES (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Create simple evaluation plot
    print("\nCreating evaluation plots...")
    
    plt.figure(figsize=(12, 4))
    
    # ROC curves
    plt.subplot(1, 3, 1)
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred_proba)
    
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC={rf_auc:.3f})')
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC={xgb_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature importance
    plt.subplot(1, 3, 2)
    top_features = feature_importance.head(8)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    
    # Confusion matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_test, rf_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Random Forest)')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('plots/quick_ml_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final summary
    print("\nSUCCESS! ML MODELS TRAINED")
    print("=" * 50)
    print(f"   Best performing model: {best_model}")
    print(f"   AUC Score: {best_auc:.4f}")
    
    if best_auc > 0.85:
        print("   Excellent predictive performance!")
    elif best_auc > 0.75:
        print("   Good predictive performance!")
    else:
        print("   Moderate performance - consider more features")
    
    print(f"\nGenerated Files:")
    print(f"   plots/quick_ml_evaluation.png")
    
    return {
        'rf_model': rf,
        'xgb_model': xgb_model,
        'rf_auc': rf_auc,
        'xgb_auc': xgb_auc,
        'best_model': best_model,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    results = main()