"""
Simple Advanced ML Visualizations (Unicode-safe)
Creates 3 additional cool ML graphs
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_loader import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

# Set cool dark styling
plt.style.use('dark_background')

def create_3d_pca_plot(X_scaled, y, title="3D PCA Feature Space"):
    """Create 3D PCA visualization"""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(12, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    
    # Normal points
    normal_mask = y == 0
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], X_pca[normal_mask, 2],
              c='lightblue', s=10, alpha=0.6, label='Normal')
    
    # Anomaly points  
    anomaly_mask = y == 1
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], X_pca[anomaly_mask, 2],
              c='red', s=30, alpha=0.9, marker='X', label='Anomalies')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', color='white')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', color='white')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', color='white')
    ax.set_title(title, color='white', fontsize=16, pad=20)
    ax.legend()
    
    # Style the 3D plot
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    
    plt.savefig('plots/3d_pca_visualization.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    return pca

def create_learning_curves(X, y, title="Learning Curves Comparison"):
    """Create learning curves for multiple models"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
        'Extra Trees': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', bootstrap=False)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='black')
    colors = ['#00ff41', '#ff073a']
    
    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=3, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot with confidence bands
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.3, color=colors[i])
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.3, color=colors[i])
        
        ax.plot(train_sizes, train_mean, 'o-', color=colors[i], 
               label=f'{name} - Training', linewidth=3, markersize=6)
        ax.plot(train_sizes, val_mean, 's-', color=colors[i], 
               label=f'{name} - Validation', linewidth=3, linestyle='--', markersize=6)
        
        ax.set_title(f'{name} Learning Curve', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Examples', color='white', fontsize=12)
        ax.set_ylabel('AUC Score', color='white', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    plt.suptitle(title, color='white', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig('plots/learning_curves_comparison.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()

def create_feature_correlation_heatmap(X, feature_names, title="Feature Correlation Heatmap"):
    """Create enhanced correlation heatmap"""
    correlation_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=(12, 10), facecolor='black')
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Create heatmap
    ax = sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                     cmap='RdBu_r', center=0, square=True,
                     xticklabels=feature_names, yticklabels=feature_names,
                     cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=20)
    
    # Style the plot
    ax.tick_params(colors='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='white')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='white')
    
    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Correlation Coefficient', color='white')
    
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()

def main():
    print("ADVANCED ML VISUALIZATIONS")
    print("=" * 50)
    
    # Load data
    print("\nLoading motor sensor data...")
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    anomaly_data = loader.detect_anomalies(method='iqr')
    
    print(f"Loaded {len(anomaly_data)} samples")
    print(f"Anomaly rate: {(anomaly_data['is_anomaly'].sum() / len(anomaly_data) * 100):.2f}%")
    
    # Prepare features
    print("\nPreparing features...")
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
    
    # Prepare data
    X = anomaly_data[feature_cols].fillna(0)
    y = anomaly_data['is_anomaly'].astype(int)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Created {len(feature_cols)} features")
    print(f"Training set: {len(X_train)} samples")
    
    # Create visualizations
    print("\nCreating 3 advanced visualizations...")
    
    print("1. Creating 3D PCA visualization...")
    pca = create_3d_pca_plot(X_train_scaled[:5000], y_train.iloc[:5000], "3D Feature Space Analysis")
    
    print("2. Creating learning curves...")
    create_learning_curves(X_train_scaled, y_train, "Model Learning Curves Analysis")
    
    print("3. Creating correlation heatmap...")
    create_feature_correlation_heatmap(X_train_scaled, feature_cols, "Feature Correlation Analysis")
    
    print("\nSUCCESS! 3 Advanced visualizations created!")
    print("Generated Files:")
    print("  plots/3d_pca_visualization.png")
    print("  plots/learning_curves_comparison.png") 
    print("  plots/correlation_heatmap.png")
    
    return True

if __name__ == "__main__":
    main()