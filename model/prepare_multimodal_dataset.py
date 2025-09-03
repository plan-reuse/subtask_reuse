#!/usr/bin/env python3
"""
Multi-modal dataset preparation for binary PDDL object classification
Includes both images and text descriptions for fruits vs liquids
"""

import os
import csv
import shutil
import json
from pathlib import Path
import pandas as pd

# Dataset paths
DATASET_ROOT = "dataset/GroceryStoreDataset/dataset"
OUTPUT_DIR = "dataset/multimodal_binary_classification"

# Binary class mapping: fruits vs liquid
CLASS_MAPPING = {
    # Fruits -> grocery
    'Apple': 'grocery', 'Avocado': 'grocery', 'Banana': 'grocery',
    'Kiwi': 'grocery', 'Lemon': 'grocery', 'Lime': 'grocery', 
    'Mango': 'grocery', 'Melon': 'grocery', 'Nectarine': 'grocery',
    'Orange': 'grocery', 'Papaya': 'grocery', 'Passion-Fruit': 'grocery',
    'Peach': 'grocery', 'Pear': 'grocery', 'Pineapple': 'grocery',
    'Plum': 'grocery', 'Pomegranate': 'grocery', 'Red-Grapefruit': 'grocery',
    'Satsumas': 'grocery',
    
    # Liquids -> liquid
    'Juice': 'liquid', 'Milk': 'liquid', 'Soy-Milk': 'liquid', 'Oat-Milk': 'liquid',
    'Sour-Milk': 'liquid',
}

def load_description(description_path):
    """Load text description from file"""
    full_path = os.path.join(DATASET_ROOT, description_path.lstrip('/'))
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            return text if text else "No description available."
    except:
        return "No description available."

def load_dataset_info():
    """Load the classes.csv file with descriptions"""
    classes_file = os.path.join(DATASET_ROOT, "classes.csv")
    dataset_info = []
    
    with open(classes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coarse_class = row['Coarse Class Name (str)']
            if coarse_class in CLASS_MAPPING:
                # Load description text
                description = load_description(row['Product Description Path (str)'])
                
                dataset_info.append({
                    'class_name': row['Class Name (str)'],
                    'coarse_class': coarse_class,
                    'pddl_class': CLASS_MAPPING[coarse_class],
                    'image_path': row['Iconic Image Path (str)'],
                    'description_path': row['Product Description Path (str)'],
                    'description_text': description
                })
    
    return dataset_info

def create_multimodal_dataset():
    """Create multi-modal dataset with images + text"""
    
    # Create output directories
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'descriptions').mkdir(parents=True, exist_ok=True)
    
    # Load dataset info
    dataset_info = load_dataset_info()
    print(f"Found {len(dataset_info)} relevant items with descriptions")
    
    # Count classes
    class_counts = {'grocery': 0, 'liquid': 0}
    for item in dataset_info:
        class_counts[item['pddl_class']] += 1
    
    print(f"Class distribution: {class_counts}")
    
    # Create mapping from class name to info
    class_to_info = {item['class_name']: item for item in dataset_info}
    
    # Process training data
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    val_file = os.path.join(DATASET_ROOT, "val.txt")
    
    # Store dataset entries
    train_data = []
    val_data = []
    
    # Process train and val splits
    for split, split_file, data_list in [('train', train_file, train_data), 
                                        ('val', val_file, val_data)]:
        processed = 0
        
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) >= 3:
                    img_path = parts[0]
                    img_basename = os.path.basename(img_path)
                    
                    # Find matching item by checking class name in filename
                    found_item = None
                    for class_name, item in class_to_info.items():
                        if img_basename.startswith(class_name):
                            found_item = item
                            break
                    
                    if found_item:
                        # Copy image
                        src_img = os.path.join(DATASET_ROOT, img_path)
                        dst_img = output_path / split / 'images' / img_basename
                        
                        if os.path.exists(src_img):
                            shutil.copy2(src_img, dst_img)
                            
                            # Create unique text filename
                            text_filename = img_basename.replace('.jpg', '.txt')
                            dst_text = output_path / split / 'descriptions' / text_filename
                            
                            # Save description text
                            with open(dst_text, 'w', encoding='utf-8') as f:
                                f.write(found_item['description_text'])
                            
                            # Add to dataset list
                            data_list.append({
                                'image_path': f'images/{img_basename}',
                                'text_path': f'descriptions/{text_filename}',
                                'label': found_item['pddl_class'],           # Binary: grocery/liquid
                                'class_name': found_item['class_name'],     # Fine-grained: Golden-Delicious
                                'coarse_class': found_item['coarse_class'], # Medium: Apple
                                'category': 'Fruit' if found_item['pddl_class'] == 'grocery' else 'Packages',
                                'description': found_item['description_text']
                            })
                            
                            processed += 1
        
        print(f"Processed {processed} items for {split} split")
        
        # Save dataset manifest for this split
        with open(output_path / f'{split}_manifest.json', 'w') as f:
            json.dump(data_list, f, indent=2)
    
    # Create overall dataset metadata
    metadata = {
        'classes': ['grocery', 'liquid'],
        'class_mapping': CLASS_MAPPING,
        'total_items': len(dataset_info),
        'class_counts': class_counts,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'description': 'Multi-modal binary classification: fruits (grocery) vs liquid items',
        'data_format': {
            'images': 'JPG format, stored in images/ subdirectory',
            'descriptions': 'Text files, stored in descriptions/ subdirectory',
            'manifests': 'JSON files linking images, text, and labels'
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create combined dataset manifest
    all_data = {
        'train': train_data,
        'val': val_data,
        'classes': ['grocery', 'liquid'],
        'class_to_idx': {'grocery': 0, 'liquid': 1}
    }
    
    with open(output_path / 'dataset_manifest.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nMulti-modal dataset prepared in: {OUTPUT_DIR}")
    print("Structure:")
    print("  train/")
    print("    images/          # Training images")
    print("    descriptions/    # Training text descriptions")
    print("  val/")
    print("    images/          # Validation images")  
    print("    descriptions/    # Validation text descriptions")
    print("  train_manifest.json   # Training data links")
    print("  val_manifest.json     # Validation data links")
    print("  dataset_manifest.json # Complete dataset info")
    print("  metadata.json         # Dataset metadata")
    
    # Show sample descriptions with hierarchy
    print(f"\nSample data with hierarchy:")
    for i, item in enumerate(train_data[:3]):
        print(f"{i+1}. Hierarchy: {item['category']} -> {item['coarse_class']} -> {item['class_name']}")
        print(f"   PDDL Label: {item['label']}")
        print(f"   Description: {item['description'][:80]}...")
        print()

if __name__ == "__main__":
    create_multimodal_dataset()