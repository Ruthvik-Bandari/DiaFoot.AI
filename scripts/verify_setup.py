#!/usr/bin/env python3
"""
Setup Verification Script
==========================

This script verifies that all dependencies are correctly installed
and the project structure is properly set up.

Run this after setting up the environment:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version():
    """Check Python version."""
    print("=" * 50)
    print("Checking Python version...")
    
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("  ❌ Python 3.10+ required!")
        return False
    
    print("  ✅ Python version OK")
    return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\n" + "=" * 50)
    print("Checking PyTorch...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Check MPS (Apple Silicon)
        mps_available = torch.backends.mps.is_available()
        print(f"  MPS available (Apple Silicon): {mps_available}")
        
        # Test tensor creation
        device = "mps" if mps_available else ("cuda" if cuda_available else "cpu")
        x = torch.randn(2, 3).to(device)
        print(f"  Test tensor on {device}: OK")
        
        print("  ✅ PyTorch OK")
        return True
        
    except ImportError as e:
        print(f"  ❌ PyTorch not installed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ PyTorch error: {e}")
        return False


def check_mlx():
    """Check MLX installation (Apple Silicon only)."""
    print("\n" + "=" * 50)
    print("Checking MLX (Apple Silicon)...")
    
    try:
        import mlx.core as mx
        print(f"  MLX installed: Yes")
        
        # Test array creation
        x = mx.array([1, 2, 3])
        print(f"  Test array: {x}")
        
        print("  ✅ MLX OK")
        return True
        
    except ImportError:
        print("  ⚠️ MLX not installed (OK if not on Apple Silicon)")
        return True  # Not critical
    except Exception as e:
        print(f"  ⚠️ MLX error: {e}")
        return True  # Not critical


def check_core_packages():
    """Check core packages."""
    print("\n" + "=" * 50)
    print("Checking core packages...")
    
    packages = [
        ("numpy", "np"),
        ("cv2", "cv2"),
        ("PIL", "Image"),
        ("albumentations", "A"),
        ("segmentation_models_pytorch", "smp"),
        ("yaml", "yaml"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    
    for package_name, import_name in packages:
        try:
            __import__(package_name.split('.')[0] if '.' in package_name else package_name)
            print(f"  {package_name}: ✅")
        except ImportError:
            print(f"  {package_name}: ❌ Not installed")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """Check project directory structure."""
    print("\n" + "=" * 50)
    print("Checking project structure...")
    
    required_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/annotations",
        "models/checkpoints",
        "models/exported",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "src/evaluation",
        "src/utils",
        "notebooks",
        "scripts",
        "tests",
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "configs/config.yaml",
        "src/__init__.py",
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"  {dir_path}/: ✅")
        else:
            print(f"  {dir_path}/: ❌ Missing")
            all_ok = False
    
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  {file_path}: ✅")
        else:
            print(f"  {file_path}: ❌ Missing")
            all_ok = False
    
    return all_ok


def check_config():
    """Check configuration loading."""
    print("\n" + "=" * 50)
    print("Checking configuration...")
    
    try:
        from src.utils.config import load_config, Config
        
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        config = load_config(config_path)
        
        print(f"  Project name: {config.get('project', {}).get('name', 'N/A')}")
        print(f"  Dataset: {config.get('dataset', {}).get('name', 'N/A')}")
        print(f"  Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
        
        print("  ✅ Configuration OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def check_gpu_memory():
    """Check GPU memory."""
    print("\n" + "=" * 50)
    print("Checking GPU memory...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024 ** 3)
            print(f"  CUDA GPU: {props.name}")
            print(f"  Total memory: {total_memory:.2f} GB")
        elif torch.backends.mps.is_available():
            # MPS doesn't expose memory info easily
            print("  MPS (Apple Silicon): Available")
            print("  Note: Memory shared with system RAM")
        else:
            print("  No GPU available, will use CPU")
        
        print("  ✅ GPU check complete")
        return True
        
    except Exception as e:
        print(f"  ⚠️ GPU check error: {e}")
        return True


def main():
    """Run all checks."""
    print("\n")
    print("=" * 50)
    print("   DiaFootAI Setup Verification")
    print("=" * 50)
    
    results = {
        "Python version": check_python_version(),
        "PyTorch": check_pytorch(),
        "MLX": check_mlx(),
        "Core packages": check_core_packages(),
        "Project structure": check_project_structure(),
        "Configuration": check_config(),
        "GPU memory": check_gpu_memory(),
    }
    
    print("\n")
    print("=" * 50)
    print("   Summary")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All checks passed! Ready to start.")
        print("\nNext steps:")
        print("  1. Download datasets: python scripts/download_dataset.py")
        print("  2. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
        return 0
    else:
        print("⚠️ Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
