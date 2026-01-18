#!/usr/bin/env python3
"""
DiaFootAI Dataset Download Script
===================================

This script downloads ALL publicly available diabetic foot ulcer and chronic wound
datasets for training, validation, and testing.

DATASETS INCLUDED:
==================

1. FUSeg 2021 (1,210 images) - Segmentation
   Source: https://github.com/uwm-bigdata/wound-segmentation
   Access: Open

2. AZH Wound Dataset (1,109 images) - Segmentation  
   Source: https://github.com/uwm-bigdata/wound-segmentation
   Access: Open

3. AZH Classification Dataset (730 images) - 4-class classification
   Source: https://github.com/uwm-bigdata/wound-classification-using-images-and-locations
   Access: Open

4. AZH Localization Dataset (1,010 images) - Detection + Classification
   Source: https://github.com/uwm-bigdata/wound_localization
   Access: Open

5. Medetec Wound Database (~592 images) - Various wound types
   Source: http://www.medetec.co.uk/files/medetec-image-databases.html
   Access: Open (manual download required)

6. Kaggle DFU Dataset (518 images) - Classification
   Source: https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu
   Access: Open (Kaggle account required)

7. Chronic Wound Database (188 cases) - Multimodal
   Source: https://chronicwounddatabase.eu/
   Access: Open (registration required)

8. WoundTissue Dataset (147 images) - 6-class tissue classification
   Source: https://github.com/akabircs/WoundTissue
   Access: Open

9. DFUC 2020/2022 (4,000+ images each) - Detection/Segmentation
   Source: https://dfu-challenge.github.io/
   Access: License agreement required (instructions provided)

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset fuseg
    python scripts/download_datasets.py --list

Author: Ruthvik
Date: 2025
"""

import os
import sys
import argparse
import shutil
import zipfile
import tarfile
import subprocess
from pathlib import Path
from typing import Optional, List
import urllib.request
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# Dataset Configuration
# ============================================

DATASETS = {
    "fuseg": {
        "name": "FUSeg 2021 - Foot Ulcer Segmentation Challenge",
        "description": "1,210 foot ulcer images with segmentation masks from 889 patients",
        "url": "https://github.com/uwm-bigdata/wound-segmentation",
        "type": "git",
        "size": "~200 MB",
        "task": "Segmentation",
        "images": 1210,
        "access": "open",
        "subdir": "data/Foot Ulcer Segmentation Challenge",
    },
    "azh_segmentation": {
        "name": "AZH Wound Segmentation Dataset",
        "description": "1,109 foot ulcer images with segmentation masks",
        "url": "https://github.com/uwm-bigdata/wound-segmentation",
        "type": "git",
        "size": "~150 MB",
        "task": "Segmentation",
        "images": 1109,
        "access": "open",
        "subdir": "data/wound_dataset",
    },
    "azh_classification": {
        "name": "AZH Wound Classification Dataset",
        "description": "730 wound images classified into 4 types: diabetic, pressure, surgical, venous",
        "url": "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations",
        "type": "git",
        "size": "~100 MB",
        "task": "Classification",
        "images": 730,
        "access": "open",
        "subdir": "data",
    },
    "azh_localization": {
        "name": "AZH Wound Localization Dataset",
        "description": "1,010 wound images with bounding boxes and body location labels",
        "url": "https://github.com/uwm-bigdata/wound_localization",
        "type": "git",
        "size": "~120 MB",
        "task": "Detection + Classification",
        "images": 1010,
        "access": "open",
        "subdir": "data",
    },
    "wound_tissue": {
        "name": "WoundTissue Dataset",
        "description": "147 wound images with 6-class tissue segmentation (granulation, necrosis, slough, maceration, tendon, bone)",
        "url": "https://github.com/akabircs/WoundTissue",
        "type": "git",
        "size": "~50 MB",
        "task": "Tissue Classification/Segmentation",
        "images": 147,
        "access": "open",
        "subdir": "data",
    },
    "medetec": {
        "name": "Medetec Wound Database",
        "description": "~592 images of various wound types (diabetic, pressure, venous ulcers)",
        "url": "http://www.medetec.co.uk/files/medetec-image-databases.html",
        "type": "manual",
        "size": "~100 MB",
        "task": "Classification",
        "images": 592,
        "access": "open",
        "instructions": """
MANUAL DOWNLOAD REQUIRED:
1. Visit: http://www.medetec.co.uk/files/medetec-image-databases.html
2. Download the wound image categories you need
3. Extract to: data/raw/medetec/
        """,
    },
    "kaggle_dfu": {
        "name": "Kaggle DFU Dataset",
        "description": "518 diabetic foot images (440 normal, 78 ulcer)",
        "url": "https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu",
        "type": "kaggle",
        "size": "~80 MB",
        "task": "Classification",
        "images": 518,
        "access": "kaggle_account",
        "kaggle_dataset": "laithjj/diabetic-foot-ulcer-dfu",
        "instructions": """
KAGGLE DOWNLOAD:
1. Install Kaggle CLI: pip install kaggle
2. Setup API: Place kaggle.json in ~/.kaggle/
3. Run: kaggle datasets download -d laithjj/diabetic-foot-ulcer-dfu
4. Or use this script with --dataset kaggle_dfu
        """,
    },
    "chronic_wound_db": {
        "name": "Chronic Wound Database (Poland)",
        "description": "188 multimodal cases (RGB + Thermal + Depth)",
        "url": "https://chronicwounddatabase.eu/",
        "type": "manual",
        "size": "~500 MB",
        "task": "Segmentation (Multimodal)",
        "images": 188,
        "access": "registration",
        "instructions": """
MANUAL DOWNLOAD REQUIRED:
1. Visit: https://chronicwounddatabase.eu/
2. Register for access
3. Download the dataset
4. Extract to: data/raw/chronic_wound_db/
        """,
    },
    "dfuc": {
        "name": "DFUC 2020/2022/2024 Datasets",
        "description": "Official MICCAI challenge datasets (4,000+ images each)",
        "url": "https://dfu-challenge.github.io/",
        "type": "license",
        "size": "~2 GB total",
        "task": "Detection, Segmentation",
        "images": "4000+ per challenge",
        "access": "license_agreement",
        "instructions": """
LICENSE AGREEMENT REQUIRED:
1. Visit: https://dfu-challenge.github.io/
2. Click "Apply for Datasets": http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php
3. Fill out the license agreement form
4. You will receive download links via email (usually within 1-2 business days)
5. Extract to: data/raw/dfuc/

This is the HIGHEST QUALITY dataset available - highly recommended!
        """,
    },
}


# ============================================
# Download Functions
# ============================================

def run_command(cmd: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return False


def download_git_repo(url: str, dest: Path, subdir: Optional[str] = None) -> bool:
    """Clone a git repository."""
    repo_name = url.split("/")[-1]
    clone_path = dest / repo_name
    
    if clone_path.exists():
        print(f"  Repository already exists at {clone_path}")
        print(f"  Pulling latest changes...")
        return run_command(["git", "pull"], cwd=clone_path)
    
    print(f"  Cloning {url}...")
    success = run_command(["git", "clone", "--depth", "1", url, str(clone_path)])
    
    if success and subdir:
        print(f"  Data located in: {clone_path / subdir}")
    
    return success


def download_kaggle_dataset(dataset_id: str, dest: Path) -> bool:
    """Download a Kaggle dataset."""
    try:
        import kaggle
        print(f"  Downloading {dataset_id} from Kaggle...")
        kaggle.api.dataset_download_files(dataset_id, path=str(dest), unzip=True)
        return True
    except ImportError:
        print("  Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        print("  Make sure ~/.kaggle/kaggle.json is configured correctly")
        return False


def download_url(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"  Downloading from {url}...")
        filename = url.split("/")[-1]
        filepath = dest / filename
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def extract_archive(filepath: Path, dest: Path) -> bool:
    """Extract a zip or tar archive."""
    try:
        if filepath.suffix == ".zip":
            with zipfile.ZipFile(filepath, 'r') as zf:
                zf.extractall(dest)
        elif filepath.suffix in [".tar", ".gz", ".tgz"]:
            with tarfile.open(filepath, 'r:*') as tf:
                tf.extractall(dest)
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


# ============================================
# Main Download Functions
# ============================================

def download_dataset(dataset_key: str, data_dir: Path) -> bool:
    """Download a specific dataset."""
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return False
    
    dataset = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset['name']}")
    print(f"{'='*60}")
    print(f"Description: {dataset['description']}")
    print(f"Size: {dataset['size']}")
    print(f"Task: {dataset['task']}")
    print(f"Access: {dataset['access']}")
    
    dest = data_dir / "raw" / dataset_key
    dest.mkdir(parents=True, exist_ok=True)
    
    if dataset["type"] == "git":
        success = download_git_repo(dataset["url"], dest, dataset.get("subdir"))
    elif dataset["type"] == "kaggle":
        success = download_kaggle_dataset(dataset["kaggle_dataset"], dest)
    elif dataset["type"] == "url":
        success = download_url(dataset["url"], dest)
    elif dataset["type"] in ["manual", "license"]:
        print(dataset["instructions"])
        success = True  # Manual download
    else:
        print(f"  Unknown download type: {dataset['type']}")
        success = False
    
    if success:
        print(f"  ✅ Dataset ready at: {dest}")
    else:
        print(f"  ❌ Download failed")
    
    return success


def download_all_open_datasets(data_dir: Path) -> dict:
    """Download all datasets with open access."""
    results = {}
    
    open_datasets = [
        "fuseg",
        "azh_segmentation", 
        "azh_classification",
        "azh_localization",
        "wound_tissue",
    ]
    
    for dataset_key in open_datasets:
        results[dataset_key] = download_dataset(dataset_key, data_dir)
    
    # Print manual download instructions
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED FOR:")
    print("="*60)
    
    manual_datasets = ["medetec", "kaggle_dfu", "chronic_wound_db", "dfuc"]
    for dataset_key in manual_datasets:
        dataset = DATASETS[dataset_key]
        print(f"\n{dataset['name']}:")
        print(dataset.get("instructions", "See website for download instructions"))
        results[dataset_key] = "manual"
    
    return results


def list_datasets():
    """List all available datasets."""
    print("\n" + "="*80)
    print("AVAILABLE DATASETS FOR DIABETIC FOOT WOUND ANALYSIS")
    print("="*80)
    
    total_images = 0
    
    for key, dataset in DATASETS.items():
        print(f"\n[{key}]")
        print(f"  Name: {dataset['name']}")
        print(f"  Images: {dataset['images']}")
        print(f"  Task: {dataset['task']}")
        print(f"  Size: {dataset['size']}")
        print(f"  Access: {dataset['access']}")
        print(f"  URL: {dataset['url']}")
        
        if isinstance(dataset['images'], int):
            total_images += dataset['images']
    
    print("\n" + "="*80)
    print(f"TOTAL AVAILABLE: ~{total_images:,}+ images")
    print("="*80)
    
    print("\nRECOMMENDED DOWNLOAD ORDER:")
    print("1. FUSeg (open) - Best segmentation dataset")
    print("2. AZH datasets (open) - Good variety")
    print("3. DFUC (license) - Largest and highest quality")
    print("4. Kaggle DFU (kaggle account) - Quick classification data")
    print("5. Medetec (manual) - Various wound types")


def create_dataset_info(data_dir: Path):
    """Create a JSON file with dataset information."""
    info = {
        "datasets": DATASETS,
        "data_directory": str(data_dir),
        "total_estimated_images": sum(
            d["images"] for d in DATASETS.values() 
            if isinstance(d["images"], int)
        ),
    }
    
    info_path = data_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nDataset info saved to: {info_path}")


# ============================================
# Utility Functions for Dataset Organization
# ============================================

def organize_downloaded_data(data_dir: Path):
    """
    Organize downloaded datasets into a unified structure.
    
    Target structure:
        data/processed/
            segmentation/
                images/
                masks/
            classification/
                diabetic/
                pressure/
                venous/
                surgical/
            tissue/
                granulation/
                slough/
                necrotic/
                ...
    """
    print("\n" + "="*60)
    print("Organizing datasets...")
    print("="*60)
    
    processed_dir = data_dir / "processed"
    
    # Create directory structure
    dirs_to_create = [
        processed_dir / "segmentation" / "images",
        processed_dir / "segmentation" / "masks",
        processed_dir / "classification" / "diabetic",
        processed_dir / "classification" / "pressure",
        processed_dir / "classification" / "venous",
        processed_dir / "classification" / "surgical",
        processed_dir / "tissue" / "granulation",
        processed_dir / "tissue" / "slough",
        processed_dir / "tissue" / "necrotic",
        processed_dir / "tissue" / "epithelial",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created.")
    print("\nNOTE: Run the data preprocessing script to populate these directories.")
    print("      python scripts/preprocess_data.py")


# ============================================
# Main Entry Point
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Download diabetic foot wound datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset fuseg
    python scripts/download_datasets.py --dataset fuseg azh_segmentation
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available datasets"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all open-access datasets"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific dataset(s) to download"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Data directory (default: project/data)"
    )
    
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize downloaded data into unified structure"
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        list_datasets()
        return 0
    
    if args.all:
        results = download_all_open_datasets(args.data_dir)
        create_dataset_info(args.data_dir)
        
        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        for dataset, status in results.items():
            if status == True:
                print(f"  ✅ {dataset}")
            elif status == "manual":
                print(f"  ⚠️  {dataset} (manual download required)")
            else:
                print(f"  ❌ {dataset}")
        return 0
    
    if args.dataset:
        for dataset_key in args.dataset:
            download_dataset(dataset_key, args.data_dir)
        return 0
    
    if args.organize:
        organize_downloaded_data(args.data_dir)
        return 0
    
    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
