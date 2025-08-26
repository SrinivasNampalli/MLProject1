import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from matplotlib.widgets import Slider, Button
import glob
import os

class Interactive3DAnalyzer:
    def __init__(self):
        self.data = None
        self.fig = None
        self.ax = None
        self.scatter = None
        self.current_features = ['position', 'temperature', 'voltage']
        self.available_features = []
        
    def load_data(self, data_path=None):
        """Load motor data from CSV files"""
        if data_path is None:
            # Load from the first available dataset
            data_path = r"C:\Users\srini\sriniprojects\MLProject1-1\data\raw\testing_data\20240527_094865"
        
        all_data = []
        motor_files = glob.glob(os.path.join(data_path, "data_motor_*.csv"))
        
        for i, file in enumerate(motor_files):
            df = pd.read_csv(file)
            df['motor_id'] = i + 1  # Add motor identifier
            all_data.append(df)
        
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Remove rows with empty labels and clean data
        self.data = self.data.dropna()
        
        # Get available numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if 'motor_id' in numeric_cols:
            numeric_cols.remove('motor_id')
        
        self.available_features = numeric_cols
        print(f"Loaded data shape: {self.data.shape}")
        print(f"Available features: {self.available_features}")
        
        return self.data
    
    def create_interactive_3d_plot(self):
        """Create interactive 3D scatter plot"""
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        # Create figure with subplots for controls
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('Interactive 3D Feature Analysis - Motor Data', fontsize=16)
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initial plot
        self.update_plot()
        
        # Add interactivity
        self.add_controls()
        
        plt.tight_layout()
        plt.show()
    
    def update_plot(self):
        """Update the 3D plot with current feature selection"""
        if self.scatter:
            self.scatter.remove()
        
        x_feature, y_feature, z_feature = self.current_features
        
        # Get data for plotting
        x = self.data[x_feature].values
        y = self.data[y_feature].values  
        z = self.data[z_feature].values
        
        # Color by motor_id if available
        if 'motor_id' in self.data.columns:
            colors = self.data['motor_id'].values
            self.scatter = self.ax.scatter(x, y, z, c=colors, cmap='viridis', 
                                         alpha=0.6, s=20)
        else:
            self.scatter = self.ax.scatter(x, y, z, alpha=0.6, s=20)
        
        # Set labels and title
        self.ax.set_xlabel(f'{x_feature}', fontsize=12)
        self.ax.set_ylabel(f'{y_feature}', fontsize=12)
        self.ax.set_zlabel(f'{z_feature}', fontsize=12)
        
        # Add colorbar if colored by motor
        if 'motor_id' in self.data.columns and not hasattr(self, 'cbar'):
            self.cbar = plt.colorbar(self.scatter, ax=self.ax, shrink=0.8)
            self.cbar.set_label('Motor ID', fontsize=10)
        
        self.fig.canvas.draw()
    
    def add_controls(self):
        """Add interactive controls for feature selection"""
        # Note: This is a simplified version. For full interactivity, 
        # you might want to use widgets in Jupyter or create separate buttons
        
        def on_key(event):
            """Handle keyboard events for feature cycling"""
            if event.key == '1':  # Change X-axis feature
                current_idx = self.available_features.index(self.current_features[0])
                next_idx = (current_idx + 1) % len(self.available_features)
                self.current_features[0] = self.available_features[next_idx]
                print(f"X-axis: {self.current_features[0]}")
                self.update_plot()
            elif event.key == '2':  # Change Y-axis feature
                current_idx = self.available_features.index(self.current_features[1])
                next_idx = (current_idx + 1) % len(self.available_features)
                self.current_features[1] = self.available_features[next_idx]
                print(f"Y-axis: {self.current_features[1]}")
                self.update_plot()
            elif event.key == '3':  # Change Z-axis feature
                current_idx = self.available_features.index(self.current_features[2])
                next_idx = (current_idx + 1) % len(self.available_features)
                self.current_features[2] = self.available_features[next_idx]
                print(f"Z-axis: {self.current_features[2]}")
                self.update_plot()
            elif event.key == 'r':  # Reset view
                self.ax.view_init(elev=20, azim=45)
                self.fig.canvas.draw()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Add instructions
        instruction_text = """
        Interactive Controls:
        - Mouse: Rotate and zoom the 3D plot
        - Key '1': Cycle X-axis feature
        - Key '2': Cycle Y-axis feature  
        - Key '3': Cycle Z-axis feature
        - Key 'r': Reset view angle
        
        Current features: X={}, Y={}, Z={}
        """.format(*self.current_features)
        
        self.fig.text(0.02, 0.02, instruction_text, fontsize=10, 
                     verticalalignment='bottom', bbox=dict(boxstyle='round', 
                     facecolor='wheat', alpha=0.8))

def create_advanced_3d_analysis():
    """Create advanced 3D analysis with multiple visualization options"""
    
    # Load and prepare data
    analyzer = Interactive3DAnalyzer()
    data = analyzer.load_data()
    
    # Create multiple 3D visualizations
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Advanced 3D Feature Analysis - Motor Data', fontsize=20)
    
    # Plot 1: Position vs Temperature vs Voltage
    ax1 = fig.add_subplot(221, projection='3d')
    if 'motor_id' in data.columns:
        scatter1 = ax1.scatter(data['position'], data['temperature'], data['voltage'], 
                              c=data['motor_id'], cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter1, ax=ax1, shrink=0.8, label='Motor ID')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Temperature') 
    ax1.set_zlabel('Voltage')
    ax1.set_title('Position vs Temperature vs Voltage')
    
    # Plot 2: Time-based analysis
    ax2 = fig.add_subplot(222, projection='3d')
    if 'time' in data.columns:
        scatter2 = ax2.scatter(data['time'], data['position'], data['temperature'],
                              c=data['voltage'], cmap='plasma', alpha=0.6, s=30)
        plt.colorbar(scatter2, ax=ax2, shrink=0.8, label='Voltage')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.set_zlabel('Temperature')
    ax2.set_title('Temporal Analysis')
    
    # Plot 3: Feature correlation in 3D
    ax3 = fig.add_subplot(223, projection='3d')
    # Create a correlation-based visualization
    features = ['position', 'temperature', 'voltage']
    for i, motor in enumerate(data['motor_id'].unique()):
        motor_data = data[data['motor_id'] == motor]
        ax3.scatter(motor_data['position'], motor_data['temperature'], 
                   motor_data['voltage'], label=f'Motor {motor}', alpha=0.7, s=25)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Temperature')
    ax3.set_zlabel('Voltage')
    ax3.set_title('Motor Comparison')
    ax3.legend()
    
    # Plot 4: Statistical analysis
    ax4 = fig.add_subplot(224, projection='3d')
    # Group data by motor and show means/distributions
    motor_stats = data.groupby('motor_id')[['position', 'temperature', 'voltage']].agg(['mean', 'std'])
    
    for motor in motor_stats.index:
        pos_mean = motor_stats.loc[motor, ('position', 'mean')]
        temp_mean = motor_stats.loc[motor, ('temperature', 'mean')]
        volt_mean = motor_stats.loc[motor, ('voltage', 'mean')]
        
        pos_std = motor_stats.loc[motor, ('position', 'std')]
        temp_std = motor_stats.loc[motor, ('temperature', 'std')]
        volt_std = motor_stats.loc[motor, ('voltage', 'std')]
        
        # Plot mean as large point
        ax4.scatter(pos_mean, temp_mean, volt_mean, s=200, alpha=0.8, 
                   label=f'Motor {motor} Mean')
        
        # Plot std as error bars (simplified)
        ax4.plot([pos_mean-pos_std, pos_mean+pos_std], [temp_mean, temp_mean], 
                [volt_mean, volt_mean], alpha=0.3)
        ax4.plot([pos_mean, pos_mean], [temp_mean-temp_std, temp_mean+temp_std], 
                [volt_mean, volt_mean], alpha=0.3)
        ax4.plot([pos_mean, pos_mean], [temp_mean, temp_mean], 
                [volt_mean-volt_std, volt_mean+volt_std], alpha=0.3)
    
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Temperature') 
    ax4.set_zlabel('Voltage')
    ax4.set_title('Statistical Summary by Motor')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, data

if __name__ == "__main__":
    # Create basic interactive analyzer
    print("Creating Interactive 3D Feature Analyzer...")
    analyzer = Interactive3DAnalyzer()
    analyzer.load_data()
    
    # Create the interactive plot
    print("Launching interactive 3D plot...")
    analyzer.create_interactive_3d_plot()
    
    # Also create advanced analysis
    print("\nCreating advanced 3D analysis...")
    fig, data = create_advanced_3d_analysis()
    
    print("\nAnalysis complete! You can now interact with the 3D plots.")
    print("The plots will remain open for manipulation.")