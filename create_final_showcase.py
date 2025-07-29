"""
Final Showcase - All graphs in one epic display
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_showcase():
    # Check which plots exist
    plot_files = [
        'plots/quick_ml_evaluation.png',
        'plots/3d_pca_visualization.png', 
        'plots/enhanced_feature_importance.png',
        'plots/learning_curves_comparison.png'
    ]
    
    existing_plots = [f for f in plot_files if os.path.exists(f)]
    
    if len(existing_plots) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='black')
        fig.suptitle('🚀 EPIC ML VISUALIZATIONS SHOWCASE 🚀', 
                     color='white', fontsize=24, fontweight='bold', y=0.95)
        
        # Load and display each plot
        for i, plot_file in enumerate(existing_plots[:4]):
            row = i // 2
            col = i % 2
            
            try:
                img = mpimg.imread(plot_file)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                
                # Add titles
                if 'quick_ml_evaluation' in plot_file:
                    axes[row, col].set_title('Enhanced ROC Curves & Confusion Matrix', 
                                           color='white', fontsize=14, pad=10)
                elif '3d_pca' in plot_file:
                    axes[row, col].set_title('3D Feature Space Analysis', 
                                           color='white', fontsize=14, pad=10)
                elif 'feature_importance' in plot_file:
                    axes[row, col].set_title('Feature Importance Analysis', 
                                           color='white', fontsize=14, pad=10)
                elif 'learning_curves' in plot_file:
                    axes[row, col].set_title('Learning Curves Comparison', 
                                           color='white', fontsize=14, pad=10)
                    
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Plot not available\n{plot_file}', 
                                  transform=axes[row, col].transAxes, 
                                  ha='center', va='center', color='white')
                axes[row, col].set_facecolor('black')
        
        # Hide empty subplots
        for i in range(len(existing_plots), 4):
            row = i // 2
            col = i % 2
            axes[row, col].axis('off')
            axes[row, col].set_facecolor('black')
        
        plt.tight_layout()
        plt.savefig('plots/EPIC_ML_SHOWCASE.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
        print("🎉 EPIC ML SHOWCASE CREATED!")
        print(f"Combined {len(existing_plots)} visualizations into one epic display!")
        return True
    else:
        print("Need at least 2 plots to create showcase")
        return False

if __name__ == "__main__":
    create_showcase()