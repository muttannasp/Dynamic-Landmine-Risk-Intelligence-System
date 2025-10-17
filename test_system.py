#!/usr/bin/env python3
"""
Test script for Dynamic Landmine Risk Intelligence System
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI
        print("✅ fastapi imported successfully")
    except ImportError as e:
        print(f"❌ fastapi import failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation"""
    print("\n🧪 Testing data generation...")
    
    try:
        from simulate_data import generate_synthetic_geodata
        df = generate_synthetic_geodata(n_points=100, seed=42)
        print(f"✅ Generated {len(df)} data points")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Mine distribution: {df['mine'].value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_model_training():
    """Test model training"""
    print("\n🧪 Testing model training...")
    
    try:
        from model_and_utils import train_rf
        from simulate_data import generate_synthetic_geodata
        
        df = generate_synthetic_geodata(n_points=200, seed=42)
        model, metrics, feature_importances = train_rf(df)
        
        print(f"✅ Model trained successfully")
        print(f"   Test AUC: {metrics['test_auc']:.3f}")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"   Feature importances: {dict(feature_importances.head())}")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_backend():
    """Test backend functionality"""
    print("\n🧪 Testing backend...")
    
    try:
        # Import backend modules
        from backend import app, startup_event
        print("✅ Backend imports successfully")
        
        # Test if we can create the app
        print(f"✅ FastAPI app created: {app.title}")
        return True
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Dynamic Landmine Risk Intelligence System - System Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_generation,
        test_model_training,
        test_backend
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("\n🚀 To start the system:")
        print("   Backend:  uvicorn backend:app --host 0.0.0.0 --port 8000 --reload")
        print("   Frontend: cd frontend && npm start")
        print("   Or use:   ./start_services.sh")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
