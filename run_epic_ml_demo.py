"""
🚀 EPIC ML DEMO LAUNCHER
Launches both the enhanced quick demo and advanced visualizations
"""

import os
import sys

def main():
    print("=" * 70)
    print("🚀 EPIC MACHINE LEARNING VISUALIZATION DEMO")
    print("=" * 70)
    print("This will create 7+ amazing ML visualizations including:")
    print("  🎨 Enhanced ROC curves with neon styling")
    print("  🌌 Interactive 3D feature space exploration")
    print("  📈 Learning curves dashboard")
    print("  🎯 Decision boundary analysis")
    print("  🔍 Advanced anomaly clustering")
    print("  🧠 Model interpretability dashboard")  
    print("  ⚡ Real-time monitoring simulation")
    print("=" * 70)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Run the enhanced quick ML demo
    print("\n🎬 Running Enhanced Quick ML Demo...")
    try:
        from quick_ml_demo import main as run_quick_demo
        results = run_quick_demo()
        print("✅ Enhanced Quick ML Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running quick demo: {e}")
        print("Trying to run advanced visualizations separately...")
        
        # Try running advanced visualizations separately
        try:
            from advanced_ml_visualizer import main as run_advanced_viz
            run_advanced_viz()
            print("✅ Advanced visualizations completed!")
        except Exception as e2:
            print(f"❌ Error running advanced visualizations: {e2}")
            return
    
    print("\n" + "=" * 70)
    print("🎉 ALL VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print("📁 Check the 'plots' folder for all generated files:")
    print("   🎨 Enhanced standard plots (.png files)")
    print("   🌐 Interactive plots (.html files - open in browser)")
    print("\n🌟 Your ML program is now EPIC with amazing visualizations!")
    print("=" * 70)

if __name__ == "__main__":
    main()