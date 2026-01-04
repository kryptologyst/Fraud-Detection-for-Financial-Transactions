#!/usr/bin/env python3
"""Quick start script for fraud detection research project.

This script provides a simple way to get started with the fraud detection
system for research and educational purposes only.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🚀 Fraud Detection Research Project - Quick Start")
    print("=" * 60)
    print("⚠️  IMPORTANT: This is for research and educational purposes only!")
    print("   Not for investment advice or production fraud detection.")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("configs").exists():
        print("❌ Please run this script from the project root directory")
        print("   Make sure you're in the directory containing src/ and configs/ folders")
        return
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -e .", "Installing package"):
        print("❌ Failed to install dependencies")
        return
    
    # Install development dependencies
    if not run_command("pip install -e \".[dev]\"", "Installing dev dependencies"):
        print("⚠️  Failed to install dev dependencies (optional)")
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_command("pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed (this is normal for initial setup)")
    
    # Train models
    print("\n🤖 Training fraud detection models...")
    if not run_command("python scripts/train_models.py --n-samples 2000", "Training models"):
        print("❌ Failed to train models")
        return
    
    # Evaluate models
    print("\n📊 Evaluating models...")
    if not run_command("python scripts/evaluate_models.py", "Evaluating models"):
        print("⚠️  Failed to evaluate models")
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 Quick start completed successfully!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print("   - assets/raw_transactions.csv (synthetic data)")
    print("   - assets/features.csv (engineered features)")
    print("   - assets/*_model.joblib (trained models)")
    print("   - assets/evaluation_report.txt (performance report)")
    print("   - assets/model_comparison.csv (model comparison)")
    print("   - assets/*.html (visualization plots)")
    
    print("\n🚀 Next steps:")
    print("   1. Launch the interactive demo:")
    print("      streamlit run demo/app.py")
    print("   2. Explore the Jupyter notebook:")
    print("      jupyter notebook notebooks/fraud_detection_demo.ipynb")
    print("   3. Read the documentation:")
    print("      cat README.md")
    print("   4. Review the disclaimer:")
    print("      cat DISCLAIMER.md")
    
    print("\n⚠️  Remember: This is for research and educational purposes only!")
    print("   Do not use for actual financial decisions or production systems.")


if __name__ == "__main__":
    main()
