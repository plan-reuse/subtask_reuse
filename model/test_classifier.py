#!/usr/bin/env python3
"""
Test the trained multi-modal classifier on individual samples
"""

import torch
from train_multimodal_classifier import MultiModalClassifier, test_single_prediction
import json
import os
import random
from PIL import Image

MODEL_PATH = "model/checkpoints/best_multimodal_model.pth"
DATASET_DIR = "dataset/multimodal_binary_classification"

def test_random_samples(num_samples=5):
    """Test on random samples from validation set"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load validation data
    with open(os.path.join(DATASET_DIR, 'val_manifest.json'), 'r') as f:
        val_data = json.load(f)
    
    # Select random samples
    samples = random.sample(val_data, min(num_samples, len(val_data)))
    
    print("Testing Multi-Modal PDDL Classifier")
    print("=" * 50)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nTest {i}:")
        print(f"True Class: {sample['label']}")
        print(f"Object: {sample['class_name']} ({sample['coarse_class']})")
        print(f"Description: {sample['description'][:100]}...")
        
        # Get paths
        image_path = os.path.join(DATASET_DIR, 'val', sample['image_path'])
        text_path = os.path.join(DATASET_DIR, 'val', sample['text_path'])
        
        # Load text description
        with open(text_path, 'r') as f:
            description = f.read().strip()
        
        # Make prediction
        try:
            pred_class, confidence = test_single_prediction(
                MODEL_PATH, image_path, description, device
            )
            
            status = "✓ CORRECT" if pred_class == sample['label'] else "✗ WRONG"
            print(f"Prediction: {pred_class} (confidence: {confidence:.3f}) {status}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)

def test_novel_object(image_path, description, object_name="Unknown"):
    """Test on a novel object not in training data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nTesting Novel Object: {object_name}")
    print(f"Description: {description}")
    print("-" * 40)
    
    try:
        pred_class, confidence = test_single_prediction(
            MODEL_PATH, image_path, description, device
        )
        
        print(f"Predicted PDDL Type: {pred_class}")
        print(f"Confidence: {confidence:.3f}")
        
        if confidence > 0.8:
            print("✓ High confidence prediction")
        elif confidence > 0.6:
            print("⚠ Medium confidence prediction") 
        else:
            print("⚡ Low confidence - may need human verification")
            
    except Exception as e:
        print(f"Error: {e}")

def get_model_info():
    """Display model information"""
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        print("Model Information:")
        print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"  Grocery Accuracy: {checkpoint['class_accuracies']['grocery']:.2f}%")
        print(f"  Liquid Accuracy: {checkpoint['class_accuracies']['liquid']:.2f}%")
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    else:
        print(f"Model not found at: {MODEL_PATH}")
        print("Please train the model first using: python model/train_multimodal_classifier.py")

if __name__ == "__main__":
    print("PDDL Object Classifier Test")
    print("=" * 30)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at: {MODEL_PATH}")
        print("Please train the model first!")
        exit(1)
    
    # Display model info
    get_model_info()
    
    # Test random samples
    test_random_samples(5)
    
    # Example: Test a novel object (uncomment and modify as needed)
    # test_novel_object(
    #     image_path="path/to/novel/object/image.jpg",
    #     description="A round red vegetable commonly used in cooking",
    #     object_name="Novel Tomato Variety"
    # )