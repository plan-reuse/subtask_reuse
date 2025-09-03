"""Qwen-based lightweight LLM classifier for object type inference."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


class QwenClassifier:
    """Lightweight Qwen model for object type classification."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading Qwen model: {model_name}...")
        
        # Force CPU usage to avoid CUDA issues
        self.device = "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        )
        self.model.to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def predict(self, obj_name: str) -> str:
        """Predict object type using Qwen model."""
        
        # Create a clear prompt for classification
        prompt = f"""Classify the kitchen object "{obj_name}" into exactly ONE of these categories:
- food (edible items, ingredients)
- cookware (tools that cook/transform food like pots, blenders, toasters)
- utensil (eating/serving tools like plates, bowls, glasses)
- container (storage items like fridges, cabinets, drawers)
- stovetop (heating surfaces like stoves, burners)
- region (locations/areas like countertops, corners)

Object: {obj_name}
Category:"""

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Short response
                    temperature=0.1,    # Low temperature for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the category from response
            category = self._extract_category(response, prompt)
            
            return category
            
        except Exception as e:
            print(f"Error predicting for '{obj_name}': {e}")
            return self._fallback_classify(obj_name)
    
    def _extract_category(self, response: str, prompt: str) -> str:
        """Extract category from model response."""
        
        # Remove the prompt part
        response = response.replace(prompt, "").strip()
        
        # Valid categories
        valid_categories = ['food', 'cookware', 'utensil', 'container', 'stovetop', 'region']
        
        # Look for valid categories in the response
        response_lower = response.lower()
        
        for category in valid_categories:
            if category in response_lower:
                return category
        
        # If no valid category found, try first word
        first_word = response.split()[0].lower() if response.split() else ""
        if first_word in valid_categories:
            return first_word
        
        # Fallback
        return self._fallback_classify(response)
    
    def _fallback_classify(self, obj_name: str) -> str:
        """Simple rule-based fallback."""
        name_lower = obj_name.lower()
        
        if any(word in name_lower for word in ['chicken', 'beef', 'salt', 'oil', 'cheese', 'bread', 'steak']):
            return 'food'
        elif any(word in name_lower for word in ['pot', 'pan', 'toaster', 'blender', 'mixer']):
            return 'cookware'
        elif any(word in name_lower for word in ['plate', 'bowl', 'glass', 'mug', 'spoon', 'fork']):
            return 'utensil'
        elif any(word in name_lower for word in ['fridge', 'cabinet', 'microwave', 'drawer']):
            return 'container'
        elif any(word in name_lower for word in ['stove', 'burner', 'cooktop']):
            return 'stovetop'
        elif any(word in name_lower for word in ['counter', 'corner', 'sink']):
            return 'region'
        else:
            return 'food'  # Default fallback


def test_qwen_classifier():
    """Test the Qwen classifier on various objects."""
    
    print("=== Qwen LLM Classifier Test ===\n")
    
    # Initialize classifier
    classifier = QwenClassifier()
    
    print("\n" + "="*60)
    
    # Test objects with expected results
    test_cases = [
        # Basic objects
        ("pot", "cookware"),
        ("plate", "utensil"), 
        ("chicken", "food"),
        ("fridge", "container"),
        ("stove", "stovetop"),
        ("countertop", "region"),
        
        ("toaster", "cookware"),
        ("blender", "cookware"),
        ("wine_glass", "utensil"),
        ("olive_oil", "food"),
        ("beef_steak", "food"),
        ("soup_bowl", "utensil"),
        ("spice_cabinet", "container"),
        
        # Novel objects
        ("sauce_pan", "cookware"),
        ("dinner_plate", "utensil"),
        ("chicken_breast", "food"),
        ("gas_stove", "stovetop"),
        ("kitchen_counter", "region"),
        ("cutting_board", "utensil"),
        ("coffee_maker", "cookware"),
        ("oven", "stovetop"),
    ]
    
    print("Classification Results:")
    print("-" * 70)
    print(f"{'Object':<20} {'Predicted':<12} {'Expected':<12} {'Correct':<8} {'Match'}")
    print("-" * 70)
    
    correct = 0
    total = len(test_cases)
    
    for obj_name, expected in test_cases:
        try:
            predicted = classifier.predict(obj_name)
            is_correct = predicted == expected
            correct += is_correct
            
            status = "✓" if is_correct else "✗"
            match_indicator = "PASS" if is_correct else "FAIL"
            
            print(f"{obj_name:<20} {predicted:<12} {expected:<12} {status:<8} {match_indicator}")
            
        except Exception as e:
            print(f"{obj_name:<20} ERROR       {expected:<12} ✗        FAIL")
            print(f"  Error: {e}")
    
    print("-" * 70)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    



if __name__ == "__main__":
    test_qwen_classifier()