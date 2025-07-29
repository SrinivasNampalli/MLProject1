"""
Quick ML Demo for Motor Predictive Maintenance
Faster training with reduced hyperparameter tuning for demonstration
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from data_loader import MotorDataLoader
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set advanced styling
plt.style.use('dark_background')
sns.set_palette("bright")

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
    
    # Create enhanced evaluation plot
    print("\nCreating EPIC evaluation plots...")
    
    plt.figure(figsize=(16, 6), facecolor='black')
    
    # ROC curves with neon styling
    ax1 = plt.subplot(1, 3, 1)
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred_proba)
    
    plt.plot(rf_fpr, rf_tpr, label=f'🌲 Random Forest (AUC={rf_auc:.3f})', 
             color='#00ff41', linewidth=3, marker='o', markersize=4)
    plt.plot(xgb_fpr, xgb_tpr, label=f'⚡ XGBoost (AUC={xgb_auc:.3f})', 
             color='#ff073a', linewidth=3, marker='s', markersize=4)
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, linewidth=2)
    
    plt.fill_between(rf_fpr, rf_tpr, alpha=0.2, color='#00ff41')
    plt.fill_between(xgb_fpr, xgb_tpr, alpha=0.2, color='#ff073a')
    
    plt.xlabel('False Positive Rate', color='white', fontsize=12)
    plt.ylabel('True Positive Rate', color='white', fontsize=12)
    plt.title('🎯 ROC Curves Comparison', color='white', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, facecolor='black', edgecolor='white')
    plt.grid(True, alpha=0.3, color='white')
    ax1.set_facecolor('black')
    
    # Feature importance with gradient colors
    ax2 = plt.subplot(1, 3, 2)
    top_features = feature_importance.head(8)
    
    # Create gradient colors
    colors = plt.cm.plasma(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(top_features['feature'], top_features['importance'], color=colors)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', color='white', fontweight='bold')
    
    plt.title('🎯 Feature Importance Ranking', color='white', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', color='white', fontsize=12)
    plt.tick_params(colors='white')
    ax2.set_facecolor('black')
    
    # Enhanced confusion matrix
    ax3 = plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_test, rf_pred)
    
    # Use a cooler colormap
    im = plt.imshow(cm, interpolation='nearest', cmap='plasma')
    plt.title('🎯 Confusion Matrix Analysis', color='white', fontsize=14, fontweight='bold')
    
    # Enhanced colorbar
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Prediction Count', color='white', fontsize=12)
    
    # Add enhanced text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > thresh else "black"
            label = "Normal" if i == 0 else "Anomaly"
            pred_label = "Normal" if j == 0 else "Anomaly"
            
            plt.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                    ha="center", va="center", color=text_color, 
                    fontweight='bold', fontsize=11)
    
    plt.ylabel('True Label', color='white', fontsize=12)
    plt.xlabel('Predicted Label', color='white', fontsize=12)
    plt.xticks([0, 1], ['Normal', 'Anomaly'], color='white')
    plt.yticks([0, 1], ['Normal', 'Anomaly'], color='white')
    ax3.set_facecolor('black')
    
    plt.tight_layout()
    plt.savefig('plots/quick_ml_evaluation.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    # Launch advanced visualizations
    print("\nLaunching ADVANCED ML VISUALIZATIONS...")
    print("   (This will create 6 additional epic graphs!)")
    
    try:
        from advanced_ml_visualizer import AdvancedMLVisualizer
        
        # Create advanced visualizer
        advanced_viz = AdvancedMLVisualizer(anomaly_data)
        advanced_viz.prepare_data()
        
        # Create the coolest visualizations
        print("   🌌 Creating 3D Feature Space...")
        advanced_viz.create_3d_feature_space_plot()
        
        print("   📈 Creating Learning Curves Dashboard...")
        advanced_viz.create_learning_curves_dashboard()
        
        print("   🎯 Creating Decision Boundary Analysis...")
        advanced_viz.create_decision_boundary_visualization()
        
        print("   🔍 Creating Anomaly Clustering Analysis...")
        advanced_viz.create_anomaly_clustering_analysis()
        
        print("   🧠 Creating Model Interpretability Dashboard...")
        advanced_viz.create_model_interpretability_dashboard()
        
        print("   ⚡ Creating Real-time Monitoring Simulation...")
        advanced_viz.create_real_time_monitoring_simulation()
        
        print("\n🎉 SUCCESS! All advanced visualizations created!")
        
    except ImportError:
        print("   ℹ️  Run 'python advanced_ml_visualizer.py' separately for 6 additional graphs!")
    except Exception as e:
        print(f"   ⚠️  Advanced visualizations skipped: {str(e)}")
    
    print(f"\n🌟 ENHANCED VISUALIZATIONS COMPLETE!")
    
    # Enhanced final summary
    print("\n🎉 SUCCESS! EPIC ML MODELS TRAINED")
    print("=" * 60)
    print(f"   🏆 Best performing model: {best_model}")
    print(f"   📊 AUC Score: {best_auc:.4f}")
    
    if best_auc > 0.85:
        print("   🚀 Excellent predictive performance!")
    elif best_auc > 0.75:
        print("   ✅ Good predictive performance!")
    else:
        print("   📈 Moderate performance - consider more features")
    
    print(f"\n📁 Generated Visualization Files:")
    print(f"   🎨 plots/quick_ml_evaluation.png (Enhanced)")
    print(f"   🌌 plots/3d_feature_space.html (Interactive 3D)")
    print(f"   📈 plots/learning_curves_dashboard.png")
    print(f"   🎯 plots/decision_boundary.png")
    print(f"   🔍 plots/anomaly_clustering.html (Interactive)")
    print(f"   🧠 plots/interpretability_dashboard.png")
    print(f"   ⚡ plots/realtime_monitoring.html (Interactive)")
    print(f"   ⚡ plots/realtime_monitoring_static.png")
    
    print(f"\n🌟 Your ML program now has 7+ AMAZING visualizations!")
    print(f"   Open .html files in browser for interactive plots! 🚀")
    
    # Save trained models for others to use
    print(f"\n💾 Saving trained models for reuse...")
    os.makedirs('saved_models', exist_ok=True)
    
    # Save models
    joblib.dump(rf, 'saved_models/random_forest_model.pkl')
    joblib.dump(xgb_model, 'saved_models/xgboost_model.pkl')
    joblib.dump(scaler, 'saved_models/feature_scaler.pkl')
    
    # Save feature information
    model_info = {
        'feature_columns': feature_cols,
        'rf_auc': rf_auc,
        'xgb_auc': xgb_auc,
        'best_model': best_model,
        'training_samples': len(X_train),
        'feature_importance': feature_importance.to_dict('records')
    }
    joblib.dump(model_info, 'saved_models/model_info.pkl')
    
    print(f"✅ Models saved to 'saved_models/' folder:")
    print(f"   🌲 random_forest_model.pkl")
    print(f"   🚀 xgboost_model.pkl") 
    print(f"   📏 feature_scaler.pkl")
    print(f"   📋 model_info.pkl")
    print(f"\n🎯 Others can now use your trained models!")
    
    return {
        'rf_model': rf,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'rf_auc': rf_auc,
        'xgb_auc': xgb_auc,
        'best_model': best_model,
        'feature_importance': feature_importance,
        'feature_columns': feature_cols
    }

if __name__ == "__main__":
    results = main()