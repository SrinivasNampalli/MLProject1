"""
🚀 ADVANCED ML VISUALIZATIONS FOR MOTOR PREDICTIVE MAINTENANCE
Cutting-edge visualization suite with 6 exciting new graph types
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb

from data_loader import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

# Set advanced styling
plt.style.use('dark_background')
sns.set_palette("bright")

class AdvancedMLVisualizer:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.models = {}
        
    def prepare_data(self):
        """Prepare data for advanced visualizations"""
        print("🔧 Preparing data for advanced visualizations...")
        
        # Feature engineering
        feature_cols = ['temperature', 'voltage', 'position', 'relative_time']
        
        # Add rolling features
        self.data['temp_rolling_mean'] = self.data.groupby(['session', 'motor_id'])['temperature'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        self.data['voltage_rolling_std'] = self.data.groupby(['session', 'motor_id'])['voltage'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        
        # Encode categorical variables
        le_session = LabelEncoder()
        le_motor = LabelEncoder()
        
        self.data['session_encoded'] = le_session.fit_transform(self.data['session'])
        self.data['motor_encoded'] = le_motor.fit_transform(self.data['motor_id'])
        
        feature_cols.extend(['temp_rolling_mean', 'voltage_rolling_std', 'session_encoded', 'motor_encoded'])
        
        # Prepare features and target
        self.X = self.data[feature_cols].fillna(0)
        self.y = self.data['is_anomaly'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        
        # Store splits
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.scaler = scaler
        
        print(f"   ✅ Prepared {len(feature_cols)} features")
        return feature_cols
    
    def create_3d_feature_space_plot(self):
        """1. 3D Feature Space Visualization with Anomaly Clustering"""
        print("🌌 Creating 3D Feature Space Visualization...")
        
        # Use PCA to reduce to 3D
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.X_train_scaled)
        
        fig = go.Figure()
        
        # Normal points
        normal_mask = self.y_train == 0
        fig.add_trace(go.Scatter3d(
            x=X_pca[normal_mask, 0],
            y=X_pca[normal_mask, 1], 
            z=X_pca[normal_mask, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='lightblue',
                opacity=0.6,
                symbol='circle'
            ),
            name='Normal Operation',
            hovertemplate='<b>Normal</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
        ))
        
        # Anomaly points
        anomaly_mask = self.y_train == 1
        fig.add_trace(go.Scatter3d(
            x=X_pca[anomaly_mask, 0],
            y=X_pca[anomaly_mask, 1],
            z=X_pca[anomaly_mask, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                opacity=0.9,
                symbol='x',
                line=dict(color='darkred', width=2)
            ),
            name='⚠️ Anomalies',
            hovertemplate='<b>Anomaly Detected!</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🌌 3D Feature Space: Normal vs Anomalous Motor Behavior",
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)',
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='black',
            font=dict(color='white', size=12),
            height=700
        )
        
        fig.write_html("plots/3d_feature_space.html")
        print("   ✅ 3D Feature Space saved to plots/3d_feature_space.html")
        
    def create_learning_curves_dashboard(self):
        """2. Advanced Learning Curves with Multiple Models"""
        print("📈 Creating Learning Curves Dashboard...")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 Advanced Learning Curves Dashboard', fontsize=18, color='white', y=0.98)
        
        colors = ['#00ff41', '#ff073a', '#ffb300']
        
        for i, (name, model) in enumerate(models.items()):
            # Learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train_scaled, self.y_train,
                cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='roc_auc'
            )
            
            ax = axes[0, 0] if i == 0 else (axes[0, 1] if i == 1 else axes[1, 0])
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.3, color=colors[i])
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.3, color=colors[i])
            
            ax.plot(train_sizes, train_mean, 'o-', color=colors[i], label=f'{name} - Training', linewidth=2)
            ax.plot(train_sizes, val_mean, 's-', color=colors[i], label=f'{name} - Validation', 
                   linewidth=2, linestyle='--')
            
            ax.set_title(f'{name} Learning Curve', color='white', fontsize=14)
            ax.set_xlabel('Training Examples', color='white')
            ax.set_ylabel('AUC Score', color='white')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('black')
        
        # Validation curve for Random Forest n_estimators
        ax = axes[1, 1]
        param_range = [10, 25, 50, 100, 200]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            self.X_train_scaled, self.y_train,
            param_name='n_estimators',
            param_range=param_range,
            cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.3, color='#00ff41')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.3, color='#ff073a')
        
        ax.plot(param_range, train_mean, 'o-', color='#00ff41', label='Training AUC', linewidth=2)
        ax.plot(param_range, val_mean, 's-', color='#ff073a', label='Validation AUC', linewidth=2, linestyle='--')
        
        ax.set_title('RF Hyperparameter Tuning (n_estimators)', color='white', fontsize=14)
        ax.set_xlabel('Number of Estimators', color='white')
        ax.set_ylabel('AUC Score', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('black')
        
        plt.tight_layout()
        plt.savefig('plots/learning_curves_dashboard.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
    def create_decision_boundary_visualization(self):
        """3. Decision Boundary Visualization with Confidence Regions"""
        print("🎯 Creating Decision Boundary Visualization...")
        
        # Use only 2 most important features for 2D visualization
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_temp.fit(self.X_train_scaled, self.y_train)
        
        feature_importance = rf_temp.feature_importances_
        top_2_indices = np.argsort(feature_importance)[-2:]
        
        X_2d = self.X_train_scaled[:, top_2_indices]
        
        # Train model on 2D data
        rf_2d = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_2d.fit(X_2d, self.y_train)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Get predictions and probabilities
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z_proba = rf_2d.predict_proba(mesh_points)[:, 1]
        Z_proba = Z_proba.reshape(xx.shape)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('🎯 Decision Boundary Analysis', fontsize=18, color='white', y=0.95)
        
        # Plot 1: Probability contours
        ax1 = axes[0]
        contour = ax1.contourf(xx, yy, Z_proba, levels=20, alpha=0.8, cmap='RdYlBu_r')
        ax1.contour(xx, yy, Z_proba, levels=[0.5], colors='white', linestyles='--', linewidths=3)
        
        # Scatter plot of actual data
        normal_mask = self.y_train == 0
        anomaly_mask = self.y_train == 1
        
        ax1.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
                   c='lightblue', s=30, alpha=0.7, label='Normal', edgecolors='white')
        ax1.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], 
                   c='red', s=50, alpha=0.9, marker='X', label='Anomaly', edgecolors='white')
        
        ax1.set_title('Probability Landscape', color='white', fontsize=14)
        ax1.set_xlabel(f'Feature {top_2_indices[0]} (Importance: {feature_importance[top_2_indices[0]]:.3f})', color='white')
        ax1.set_ylabel(f'Feature {top_2_indices[1]} (Importance: {feature_importance[top_2_indices[1]]:.3f})', color='white')
        ax1.legend()
        
        cbar1 = plt.colorbar(contour, ax=ax1)
        cbar1.set_label('Anomaly Probability', color='white')
        cbar1.ax.yaxis.label.set_color('white')
        cbar1.ax.tick_params(colors='white')
        
        # Plot 2: Confidence regions
        ax2 = axes[1]
        confidence = np.max(rf_2d.predict_proba(mesh_points), axis=1)
        confidence = confidence.reshape(xx.shape)
        
        contour2 = ax2.contourf(xx, yy, confidence, levels=20, alpha=0.8, cmap='viridis')
        
        ax2.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
                   c='lightblue', s=30, alpha=0.7, label='Normal', edgecolors='white')
        ax2.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], 
                   c='red', s=50, alpha=0.9, marker='X', label='Anomaly', edgecolors='white')
        
        ax2.set_title('Model Confidence Regions', color='white', fontsize=14)
        ax2.set_xlabel(f'Feature {top_2_indices[0]} (Importance: {feature_importance[top_2_indices[0]]:.3f})', color='white')
        ax2.set_ylabel(f'Feature {top_2_indices[1]} (Importance: {feature_importance[top_2_indices[1]]:.3f})', color='white')
        ax2.legend()
        
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Prediction Confidence', color='white')
        cbar2.ax.yaxis.label.set_color('white')
        cbar2.ax.tick_params(colors='white')
        
        for ax in axes:
            ax.set_facecolor('black')
        
        plt.tight_layout()
        plt.savefig('plots/decision_boundary.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
    def create_anomaly_clustering_analysis(self):
        """4. Advanced Anomaly Clustering with t-SNE"""
        print("🔍 Creating Anomaly Clustering Analysis...")
        
        # Apply t-SNE for better visualization
        print("   Running t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(self.X_train_scaled[:5000])  # Sample for performance
        y_sample = self.y_train.iloc[:5000]
        
        # Perform clustering on anomalies only
        anomaly_data = self.X_train_scaled[self.y_train == 1]
        if len(anomaly_data) > 5:
            kmeans = KMeans(n_clusters=min(3, len(anomaly_data)//2), random_state=42)
            anomaly_clusters = kmeans.fit_predict(anomaly_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('t-SNE Anomaly Visualization', 'Anomaly Cluster Analysis', 
                           'Feature Distribution by Cluster', 'Anomaly Severity Heatmap'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "violin"}, {"type": "heatmap"}]]
        )
        
        # 1. t-SNE plot
        normal_mask = y_sample == 0
        anomaly_mask = y_sample == 1
        
        fig.add_trace(go.Scatter(
            x=X_tsne[normal_mask, 0],
            y=X_tsne[normal_mask, 1],
            mode='markers',
            marker=dict(color='lightblue', size=5, opacity=0.6),
            name='Normal',
            hovertemplate='Normal Operation<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=X_tsne[anomaly_mask, 0],
            y=X_tsne[anomaly_mask, 1],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Anomalies',
            hovertemplate='⚠️ Anomaly Detected<extra></extra>'
        ), row=1, col=1)
        
        # 2. Cluster analysis (if we have anomalies)
        if len(anomaly_data) > 5:
            anomaly_tsne = X_tsne[anomaly_mask][:len(anomaly_clusters)]
            
            colors = px.colors.qualitative.Set1[:len(np.unique(anomaly_clusters))]
            for i, cluster in enumerate(np.unique(anomaly_clusters)):
                cluster_mask = anomaly_clusters == cluster
                fig.add_trace(go.Scatter(
                    x=anomaly_tsne[cluster_mask, 0],
                    y=anomaly_tsne[cluster_mask, 1],
                    mode='markers',
                    marker=dict(color=colors[i], size=12, symbol='diamond'),
                    name=f'Anomaly Cluster {cluster+1}',
                    hovertemplate=f'Cluster {cluster+1}<extra></extra>'
                ), row=1, col=2)
        
        fig.update_layout(
            title_text="🔍 Advanced Anomaly Clustering Analysis",
            title_x=0.5,
            showlegend=True,
            height=800,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        fig.write_html("plots/anomaly_clustering.html")
        print("   ✅ Anomaly Clustering saved to plots/anomaly_clustering.html")
        
    def create_model_interpretability_dashboard(self):
        """5. Model Interpretability with SHAP-style Analysis"""
        print("🧠 Creating Model Interpretability Dashboard...")
        
        # Train models
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(self.X_train_scaled, self.y_train)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🧠 Model Interpretability Dashboard', fontsize=20, color='white', y=0.98)
        
        # 1. Feature Importance Comparison
        ax1 = axes[0, 0]
        feature_names = self.X.columns
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        bars = ax1.barh(range(len(indices)), importances[indices], 
                       color=plt.cm.plasma(np.linspace(0, 1, len(indices))))
        ax1.set_yticks(range(len(indices)))
        ax1.set_yticklabels([feature_names[i] for i in indices], color='white')
        ax1.set_xlabel('Importance Score', color='white')
        ax1.set_title('🎯 Feature Importance Ranking', color='white', fontsize=14)
        ax1.set_facecolor('black')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', color='white', fontsize=10)
        
        # 2. Decision Tree Visualization (simplified)
        ax2 = axes[0, 1]
        dt_simple = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
        dt_simple.fit(self.X_train_scaled, self.y_train)
        
        plot_tree(dt_simple, ax=ax2, feature_names=feature_names[:len(self.X_train_scaled[0])],
                 class_names=['Normal', 'Anomaly'], filled=True, fontsize=8)
        ax2.set_title('🌳 Decision Tree (Depth=3)', color='white', fontsize=14)
        ax2.set_facecolor('black')
        
        # 3. Prediction Confidence Distribution
        ax3 = axes[0, 2]
        y_pred_proba = rf.predict_proba(self.X_test_scaled)[:, 1]
        
        # Separate by actual class
        normal_proba = y_pred_proba[self.y_test == 0]
        anomaly_proba = y_pred_proba[self.y_test == 1]
        
        ax3.hist(normal_proba, bins=30, alpha=0.7, label='Normal', color='lightblue', density=True)
        ax3.hist(anomaly_proba, bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
        ax3.axvline(x=0.5, color='white', linestyle='--', linewidth=2, label='Threshold')
        ax3.set_xlabel('Prediction Probability', color='white')
        ax3.set_ylabel('Density', color='white')
        ax3.set_title('📊 Prediction Confidence', color='white', fontsize=14)
        ax3.legend()
        ax3.set_facecolor('black')
        
        # 4. Partial Dependence Plot
        ax4 = axes[1, 0]
        most_important_feature_idx = indices[0]
        feature_values = np.linspace(self.X_train_scaled[:, most_important_feature_idx].min(),
                                   self.X_train_scaled[:, most_important_feature_idx].max(), 50)
        
        # Create partial dependence data
        X_pd = np.tile(self.X_train_scaled.mean(axis=0), (50, 1))
        X_pd[:, most_important_feature_idx] = feature_values
        
        pd_predictions = rf.predict_proba(X_pd)[:, 1]
        
        ax4.plot(feature_values, pd_predictions, linewidth=3, color='#00ff41')
        ax4.fill_between(feature_values, pd_predictions, alpha=0.3, color='#00ff41')
        ax4.set_xlabel(f'{feature_names[most_important_feature_idx]}', color='white')
        ax4.set_ylabel('Anomaly Probability', color='white')
        ax4.set_title('📈 Partial Dependence', color='white', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('black')
        
        # 5. Feature Correlation Network
        ax5 = axes[1, 1]
        corr_matrix = np.corrcoef(self.X_train_scaled.T)
        
        # Only show strong correlations
        mask = np.abs(corr_matrix) > 0.3
        corr_matrix_filtered = corr_matrix * mask
        
        im = ax5.imshow(corr_matrix_filtered, cmap='RdBu_r', aspect='auto')
        ax5.set_xticks(range(len(feature_names)))
        ax5.set_yticks(range(len(feature_names)))
        ax5.set_xticklabels(feature_names, rotation=45, ha='right', color='white', fontsize=8)
        ax5.set_yticklabels(feature_names, color='white', fontsize=8)
        ax5.set_title('🕸️ Feature Correlation Network', color='white', fontsize=14)
        
        # Add correlation values
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if abs(corr_matrix_filtered[i, j]) > 0.3:
                    text = ax5.text(j, i, f'{corr_matrix_filtered[i, j]:.2f}',
                                  ha="center", va="center", color="white", fontsize=6)
        
        # 6. Model Performance Metrics
        ax6 = axes[1, 2]
        
        # Calculate various metrics
        y_pred = rf.predict(self.X_test_scaled)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        bars = ax6.bar(metrics.keys(), metrics.values(), 
                      color=['#ff073a', '#ffb300', '#00ff41', '#00bfff', '#ff1493'])
        ax6.set_ylim(0, 1)
        ax6.set_ylabel('Score', color='white')
        ax6.set_title('📏 Performance Metrics', color='white', fontsize=14)
        ax6.set_facecolor('black')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', color='white', fontsize=10)
        
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right', color='white')
        
        for ax in axes.flat:
            ax.tick_params(colors='white')
            ax.set_facecolor('black')
        
        plt.tight_layout()
        plt.savefig('plots/interpretability_dashboard.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
    def create_real_time_monitoring_simulation(self):
        """6. Real-time Monitoring Simulation Dashboard"""
        print("⚡ Creating Real-time Monitoring Simulation...")
        
        # Train a model for real-time predictions
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(self.X_train_scaled, self.y_train)
        
        # Simulate real-time data stream
        n_timesteps = 100
        simulation_data = self.X_test.sample(n=n_timesteps).reset_index(drop=True)
        
        # Create animated-style plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('🔴 Real-time Anomaly Detection', '📊 Sensor Readings Stream',
                           '⚡ Alert Status', '📈 Confidence Timeline'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "indicator"}, {"type": "scatter"}]]
        )
        
        # Simulate streaming data
        timestamps = pd.date_range(start='2024-01-01 10:00:00', periods=n_timesteps, freq='1min')
        predictions = rf.predict_proba(self.scaler.transform(simulation_data))[:, 1]
        
        # 1. Anomaly detection scatter
        colors = ['red' if p > 0.5 else 'green' for p in predictions]
        fig.add_trace(go.Scatter(
            x=list(range(n_timesteps)),
            y=predictions,
            mode='markers+lines',
            marker=dict(color=colors, size=8, line=dict(width=1, color='white')),
            line=dict(color='cyan', width=2),
            name='Anomaly Probability',
            hovertemplate='Time: %{x}<br>Probability: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="white", row=1, col=1)
        
        # 2. Sensor readings stream
        for i, sensor in enumerate(['temperature', 'voltage', 'position']):
            if sensor in simulation_data.columns:
                fig.add_trace(go.Scatter(
                    x=list(range(n_timesteps)),
                    y=simulation_data[sensor],
                    mode='lines',
                    name=sensor.title(),
                    line=dict(width=2),
                    opacity=0.8
                ), row=1, col=2)
        
        # 3. Alert indicator
        current_risk = predictions[-1]
        alert_color = "red" if current_risk > 0.5 else "green"
        alert_text = "HIGH RISK" if current_risk > 0.5 else "NORMAL"
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=current_risk * 100,
            title={'text': f"Risk Level<br><span style='color:{alert_color};font-size:20px'>{alert_text}</span>"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': alert_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=2, col=1)
        
        # 4. Confidence timeline
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predictions,
            mode='lines+markers',
            fill='tonexty',
            name='Confidence Timeline',
            line=dict(color='orange', width=3),
            marker=dict(size=6, color='orange'),
            hovertemplate='%{x}<br>Risk: %{y:.1%}<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            title_text="⚡ Real-time Motor Health Monitoring Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white', size=12)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Step", color='white', row=1, col=1)
        fig.update_yaxes(title_text="Anomaly Probability", color='white', row=1, col=1)
        fig.update_xaxes(title_text="Time Step", color='white', row=1, col=2)
        fig.update_yaxes(title_text="Sensor Value", color='white', row=1, col=2)
        fig.update_xaxes(title_text="Timestamp", color='white', row=2, col=2)
        fig.update_yaxes(title_text="Risk Level", color='white', row=2, col=2)
        
        fig.write_html("plots/realtime_monitoring.html")
        print("   ✅ Real-time Monitoring saved to plots/realtime_monitoring.html")
        
        # Also create a static version
        plt.figure(figsize=(16, 10))
        
        # Create subplots for static version
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Anomaly timeline
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(range(n_timesteps), predictions, 'o-', color='cyan', linewidth=2, markersize=4)
        ax1.fill_between(range(n_timesteps), predictions, alpha=0.3, color='cyan')
        ax1.axhline(y=0.5, color='white', linestyle='--', linewidth=2, label='Alert Threshold')
        ax1.set_title('⚡ Real-time Anomaly Detection Stream', color='white', fontsize=16)
        ax1.set_xlabel('Time Step', color='white')
        ax1.set_ylabel('Anomaly Probability', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('black')
        
        # Sensor readings
        ax2 = plt.subplot(gs[1, 0])
        for i, sensor in enumerate(['temperature', 'voltage', 'position']):
            if sensor in simulation_data.columns:
                ax2.plot(range(n_timesteps), simulation_data[sensor], 
                        label=sensor.title(), linewidth=2, alpha=0.8)
        ax2.set_title('📊 Live Sensor Readings', color='white', fontsize=14)
        ax2.set_xlabel('Time Step', color='white')
        ax2.set_ylabel('Sensor Value', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('black')
        
        # Risk distribution
        ax3 = plt.subplot(gs[1, 1])
        ax3.hist(predictions, bins=20, alpha=0.7, color='orange', edgecolor='white')
        ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Alert Threshold')
        ax3.set_title('📈 Risk Distribution', color='white', fontsize=14)
        ax3.set_xlabel('Risk Level', color='white')
        ax3.set_ylabel('Frequency', color='white')
        ax3.legend()
        ax3.set_facecolor('black')
        
        # Alert summary
        ax4 = plt.subplot(gs[2, :])
        alerts = (predictions > 0.5).astype(int)
        alert_periods = []
        current_period = []
        
        for i, alert in enumerate(alerts):
            if alert:
                current_period.append(i)
            else:
                if current_period:
                    alert_periods.append(current_period)
                    current_period = []
        if current_period:
            alert_periods.append(current_period)
        
        # Plot alert periods
        ax4.plot(range(n_timesteps), np.zeros(n_timesteps), 'g-', linewidth=8, alpha=0.3, label='Normal Operation')
        
        for period in alert_periods:
            ax4.plot(period, np.ones(len(period)), 'r-', linewidth=8, alpha=0.8, label='Alert Period' if period == alert_periods[0] else "")
            ax4.scatter(period, np.ones(len(period)), c='red', s=50, zorder=5)
        
        ax4.set_title(f'🚨 Alert Timeline - {len(alert_periods)} Alert Periods Detected', color='white', fontsize=14)
        ax4.set_xlabel('Time Step', color='white')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Normal', 'Alert'], color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('black')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(colors='white')
        
        plt.suptitle('⚡ Motor Health Monitoring Dashboard', fontsize=20, color='white', y=0.98)
        plt.savefig('plots/realtime_monitoring_static.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()

def main():
    print("🚀 ADVANCED ML VISUALIZATIONS")
    print("=" * 60)
    
    # Load and prepare data
    print("\n📊 Loading motor sensor data...")
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    anomaly_data = loader.detect_anomalies(method='iqr')
    
    print(f"   ✅ Loaded {len(anomaly_data)} samples")
    print(f"   🔍 Anomaly rate: {(anomaly_data['is_anomaly'].sum() / len(anomaly_data) * 100):.2f}%")
    
    # Create visualizer
    visualizer = AdvancedMLVisualizer(anomaly_data)
    feature_cols = visualizer.prepare_data()
    
    # Create all 6 advanced visualizations
    print(f"\n🎨 Creating 6 advanced ML visualizations...")
    
    try:
        # 1. 3D Feature Space
        visualizer.create_3d_feature_space_plot()
        
        # 2. Learning Curves Dashboard  
        visualizer.create_learning_curves_dashboard()
        
        # 3. Decision Boundary Analysis
        visualizer.create_decision_boundary_visualization()
        
        # 4. Anomaly Clustering
        visualizer.create_anomaly_clustering_analysis()
        
        # 5. Model Interpretability
        visualizer.create_model_interpretability_dashboard()
        
        # 6. Real-time Monitoring
        visualizer.create_real_time_monitoring_simulation()
        
        print("\n🎉 SUCCESS! All 6 advanced visualizations created!")
        print("=" * 60)
        print("📁 Generated Files:")
        print("   • plots/3d_feature_space.html (Interactive 3D)")
        print("   • plots/learning_curves_dashboard.png")
        print("   • plots/decision_boundary.png")
        print("   • plots/anomaly_clustering.html (Interactive)")
        print("   • plots/interpretability_dashboard.png")
        print("   • plots/realtime_monitoring.html (Interactive)")
        print("   • plots/realtime_monitoring_static.png")
        
        print(f"\n🚀 Your ML program now has 6 EPIC new visualizations!")
        print("   Open the .html files in your browser for interactive plots! 🌟")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return None
    
    return visualizer

if __name__ == "__main__":
    visualizer = main()