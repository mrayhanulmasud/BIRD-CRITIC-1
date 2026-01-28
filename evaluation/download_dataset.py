#!/usr/bin/env python3
"""
Download BIRD-CRITIC dataset from HuggingFace
"""

import json
import os
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed")
    print("Please install it with: pip install datasets")
    exit(1)


def download_dataset(dataset_name: str, split: str, output_path: str):
    """Download dataset from HuggingFace and save as JSONL."""
    
    print(f"Downloading {dataset_name} (split: {split})...")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Downloaded {len(dataset)} samples")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Save as JSONL
        with open(output_path, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved to {output_path}")
        print(f"\nFirst sample preview:")
        print(json.dumps(dataset[0], indent=2)[:500] + "...")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download BIRD-CRITIC dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="birdsql/bird-critic-1.0-flash-exp",
        choices=[
            "birdsql/bird-critic-1.0-flash-exp",
            "birdsql/bird-critic-1.0-open",
            "birdsql/bird-critic-1.0-postgresql"
        ],
        help="Dataset to download"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (e.g., 'flash', 'open', 'train', 'test'). Auto-detected if not specified."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    
    # Auto-detect split and output path based on dataset
    if args.dataset == "birdsql/bird-critic-1.0-flash-exp":
        split = args.split or "flash"
        output = args.output or "data/bird-critic-flash.jsonl"
    elif args.dataset == "birdsql/bird-critic-1.0-open":
        split = args.split or "open"
        output = args.output or "data/bird-critic-open.jsonl"
    elif args.dataset == "birdsql/bird-critic-1.0-postgresql":
        split = args.split or "train"  # or "test", check available splits
        output = args.output or "data/bird-critic-postgresql.jsonl"
    else:
        split = args.split or "train"
        output = args.output or "data/dataset.jsonl"
    
    download_dataset(args.dataset, split, output)


if __name__ == "__main__":
    main()
