import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import glob
import os

def load_motor_data():
    """Load motor data from CSV files"""
    data_path = r"C:\Users\srini\sriniprojects\MLProject1-1\data\raw\testing_data\20240527_094865"
    
    all_data = []
    motor_files = glob.glob(os.path.join(data_path, "data_motor_*.csv"))
    
    for i, file in enumerate(motor_files):
        df = pd.read_csv(file)
        df['motor_id'] = i + 1
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    data = data.dropna()
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    print(f"Sample data:\n{data.head()}")
    
    return data

def create_3d_feature_plots():
    """Create multiple 3D feature analysis plots"""
    
    # Load data
    data = load_motor_data()
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('3D Feature Analysis - Motor Data', fontsize=20, fontweight='bold')
    
    # Plot 1: Position vs Temperature vs Voltage (colored by motor)
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(data['position'], data['temperature'], data['voltage'], 
                          c=data['motor_id'], cmap='tab10', alpha=0.7, s=50)
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Temperature', fontsize=12)
    ax1.set_zlabel('Voltage', fontsize=12)
    ax1.set_title('Position vs Temperature vs Voltage\n(Colored by Motor ID)', fontsize=14)
    plt.colorbar(scatter1, ax=ax1, shrink=0.6, label='Motor ID')
    
    # Plot 2: Time series in 3D
    ax2 = fig.add_subplot(232, projection='3d')
    scatter2 = ax2.scatter(data['time'], data['position'], data['temperature'],
                          c=data['voltage'], cmap='plasma', alpha=0.7, s=50)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_zlabel('Temperature', fontsize=12)
    ax2.set_title('Temporal Analysis\n(Colored by Voltage)', fontsize=14)
    plt.colorbar(scatter2, ax=ax2, shrink=0.6, label='Voltage')
    
    # Plot 3: Motor comparison
    ax3 = fig.add_subplot(233, projection='3d')
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, motor in enumerate(sorted(data['motor_id'].unique())):
        motor_data = data[data['motor_id'] == motor]
        ax3.scatter(motor_data['position'], motor_data['temperature'], 
                   motor_data['voltage'], label=f'Motor {motor}', 
                   color=colors[i % len(colors)], alpha=0.7, s=40)
    ax3.set_xlabel('Position', fontsize=12)
    ax3.set_ylabel('Temperature', fontsize=12)
    ax3.set_zlabel('Voltage', fontsize=12)
    ax3.set_title('Individual Motor Comparison', fontsize=14)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Voltage vs Time vs Position
    ax4 = fig.add_subplot(234, projection='3d')
    scatter4 = ax4.scatter(data['voltage'], data['time'], data['position'],
                          c=data['temperature'], cmap='coolwarm', alpha=0.7, s=50)
    ax4.set_xlabel('Voltage', fontsize=12)
    ax4.set_ylabel('Time', fontsize=12)
    ax4.set_zlabel('Position', fontsize=12)
    ax4.set_title('Voltage vs Time vs Position\n(Colored by Temperature)', fontsize=14)
    plt.colorbar(scatter4, ax=ax4, shrink=0.6, label='Temperature')
    
    # Plot 5: Statistical summary
    ax5 = fig.add_subplot(235, projection='3d')
    motor_stats = data.groupby('motor_id')[['position', 'temperature', 'voltage']].mean()
    
    for motor in motor_stats.index:
        pos_mean = motor_stats.loc[motor, 'position']
        temp_mean = motor_stats.loc[motor, 'temperature']
        volt_mean = motor_stats.loc[motor, 'voltage']
        
        ax5.scatter(pos_mean, temp_mean, volt_mean, s=300, alpha=0.8, 
                   label=f'Motor {motor}', color=colors[(motor-1) % len(colors)])
        
        # Add motor data cloud around mean
        motor_data = data[data['motor_id'] == motor]
        ax5.scatter(motor_data['position'], motor_data['temperature'], 
                   motor_data['voltage'], alpha=0.1, s=20, 
                   color=colors[(motor-1) % len(colors)])
    
    ax5.set_xlabel('Position', fontsize=12)
    ax5.set_ylabel('Temperature', fontsize=12)
    ax5.set_zlabel('Voltage', fontsize=12)
    ax5.set_title('Motor Centroids with Data Clouds', fontsize=14)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 6: Feature correlation visualization
    ax6 = fig.add_subplot(236, projection='3d')
    # Normalize features for better visualization
    norm_pos = (data['position'] - data['position'].min()) / (data['position'].max() - data['position'].min())
    norm_temp = (data['temperature'] - data['temperature'].min()) / (data['temperature'].max() - data['temperature'].min())
    norm_volt = (data['voltage'] - data['voltage'].min()) / (data['voltage'].max() - data['voltage'].min())
    
    scatter6 = ax6.scatter(norm_pos, norm_temp, norm_volt,
                          c=data['motor_id'], cmap='viridis', alpha=0.7, s=50)
    ax6.set_xlabel('Normalized Position', fontsize=12)
    ax6.set_ylabel('Normalized Temperature', fontsize=12)
    ax6.set_zlabel('Normalized Voltage', fontsize=12)
    ax6.set_title('Normalized Feature Space\n(All features 0-1)', fontsize=14)
    plt.colorbar(scatter6, ax=ax6, shrink=0.6, label='Motor ID')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig('C:\\Users\\srini\\sriniprojects\\MLProject1-1\\plots\\3d_feature_analysis_interactive.png', 
                dpi=300, bbox_inches='tight')
    
    print("\n3D Analysis Complete!")
    print("- Plot saved as '3d_feature_analysis_interactive.png'")
    print("- You can now rotate, zoom, and interact with each 3D subplot")
    print("- Each subplot shows different feature relationships")
    
    plt.show()
    
    return fig, data

if __name__ == "__main__":
    fig, data = create_3d_feature_plots()
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Motors: {sorted(data['motor_id'].unique())}")
    print(f"Feature ranges:")
    for col in ['position', 'temperature', 'voltage']:
        print(f"  {col}: {data[col].min():.2f} - {data[col].max():.2f}")