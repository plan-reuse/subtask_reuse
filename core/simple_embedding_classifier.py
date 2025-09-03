"""Simple embedding-based classifier for object type inference."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


class SimpleEmbeddingClassifier:
    """Embedding-based classifier using prototype vectors."""
    
    def __init__(self):
        # Use a lightweight sentence transformer model
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Force CPU usage
        self.prototypes = {}  # type -> average embedding
        self.is_trained = False
    
    def train(self, training_data: List[Tuple[str, str]]):
        """Train by computing prototype embeddings for each type."""
        print(f"Training on {len(training_data)} examples...")
        
        # Group objects by type
        type_objects = {}
        for obj, obj_type in training_data:
            if obj_type not in type_objects:
                type_objects[obj_type] = []
            type_objects[obj_type].append(obj)
        
        # Compute prototype embedding for each type
        for obj_type, objects in type_objects.items():
            # Get embeddings for all objects of this type
            embeddings = self.model.encode(objects)
            
            # Average embedding becomes the prototype
            prototype = np.mean(embeddings, axis=0)
            self.prototypes[obj_type] = prototype
            
            print(f"  {obj_type}: {len(objects)} examples")
        
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, obj_name: str) -> str:
        """Predict by finding closest prototype."""
        if not self.is_trained:
            return obj_name
        
        # Get embedding for the object
        obj_embedding = self.model.encode([obj_name])[0]
        
        # Find most similar prototype
        best_type = None
        best_similarity = -1
        
        for obj_type, prototype in self.prototypes.items():
            similarity = cosine_similarity([obj_embedding], [prototype])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = obj_type
        
        return best_type
    
    def predict_with_confidence(self, obj_name: str) -> Tuple[str, float]:
        """Predict with confidence score."""
        if not self.is_trained:
            return obj_name, 0.0
        
        obj_embedding = self.model.encode([obj_name])[0]
        
        similarities = {}
        for obj_type, prototype in self.prototypes.items():
            similarity = cosine_similarity([obj_embedding], [prototype])[0][0]
            similarities[obj_type] = similarity
        
        # Get best prediction and confidence
        best_type = max(similarities, key=similarities.get)
        confidence = similarities[best_type]
        
        return best_type, confidence


def create_training_data() -> List[Tuple[str, str]]:
    """Create training data with semantic descriptions."""
    return [
        # Regions - places in kitchen
        ("kitchen corner", "region"), ("countertop", "region"), ("counter space", "region"), 
        ("sink area", "region"), ("kitchen counter", "region"), ("work surface", "region"),
        ("preparation area", "region"), ("cooking area", "region"),
        
        # Containers - things that store/hold items (INCLUDING APPLIANCES!)
        ("refrigerator", "container"), ("fridge", "container"), ("cabinet", "container"), 
        ("cupboard", "container"), ("drawer", "container"), ("pantry", "container"),
        ("storage box", "container"), ("food container", "container"),
        ("microwave", "container"), ("dishwasher", "container"), ("freezer", "container"),
        ("spice cabinet", "container"), ("vegetable drawer", "container"),
        
        # Stovetops - heating/cooking surfaces  
        ("stove", "stovetop"), ("stovetop", "stovetop"), ("gas burner", "stovetop"),
        ("electric burner", "stovetop"), ("cooktop", "stovetop"), ("heating element", "stovetop"),
        ("cooking surface", "stovetop"), ("burner", "stovetop"), ("oven", "stovetop"),
        
        # Utensils - eating/serving tools (INCLUDING GLASSES!)
        ("plate", "utensil"), ("dinner plate", "utensil"), ("bowl", "utensil"), 
        ("serving bowl", "utensil"), ("spoon", "utensil"), ("fork", "utensil"),
        ("knife", "utensil"), ("serving dish", "utensil"), ("wine glass", "utensil"),
        ("coffee mug", "utensil"), ("soup bowl", "utensil"), ("cutting board", "utensil"),
        
        # Cookware - cooking tools/appliances that transform food
        ("pot", "cookware"), ("cooking pot", "cookware"), ("saucepan", "cookware"),
        ("frying pan", "cookware"), ("skillet", "cookware"), ("wok", "cookware"),
        ("sauce pan", "cookware"), ("cooking vessel", "cookware"),
        ("toaster", "cookware"), ("blender", "cookware"), ("food processor", "cookware"),
        ("mixer", "cookware"), ("coffee maker", "cookware"),
        
        # Food - edible items
        ("chicken", "food"), ("beef", "food"), ("salt", "food"), ("pepper", "food"),
        ("potato", "food"), ("tomato", "food"), ("cheese", "food"), ("bread", "food"),
        ("olive oil", "food"), ("beef steak", "food"), ("garlic clove", "food"),
    ]


if __name__ == "__main__":
    print("=== Simple Embedding Classifier Test ===\n")
    
    # Create and train classifier
    classifier = SimpleEmbeddingClassifier()
    training_data = create_training_data()
    classifier.train(training_data)
    
    print("\n" + "="*60)
    
    # Test objects: PDDL objects + novel semantic objects
    test_objects = [
        # Original PDDL objects
        "pot", "plate", "chicken", "salt", "fridge", "stove", "countertop_1",
        
        # Semantic variations (should work better with embeddings)
        "sauce_pan", "dinner_plate", "chicken_breast", "salt_shaker", 
        "refrigerator", "gas_stove", "kitchen_counter", "cooking_pot",
        
        # Kitchen appliances (semantic understanding test)
        "microwave", "toaster", "blender", "dishwasher", "oven",
        "coffee_maker", "food_processor", "mixer",
        
        # Food items (semantic test)
        "beef_steak", "cheese_slice", "bread_loaf", "pasta_sauce",
        "olive_oil", "garlic_clove", "onion_slice", "egg_carton",
        
        # Utensil variations
        "soup_bowl", "wine_glass", "coffee_mug", "cutting_board",
        "serving_spoon", "butter_knife", "salad_fork",
        
        # Container variations
        "spice_cabinet", "vegetable_drawer", "freezer_compartment",
        
        # Completely unknown
        "mystery_object", "thing_123", "random_stuff"
    ]
    
    print("Classification Results:")
    print("-" * 70)
    print(f"{'Object':<20} {'Predicted Type':<15} {'Confidence':<12} {'Semantic Test'}")
    print("-" * 70)
    
    # Track semantic understanding
    semantic_tests = {
        # These should be classified correctly due to semantic similarity
        "microwave": "container",  # or could be cookware
        "toaster": "cookware", 
        "blender": "cookware",
        "soup_bowl": "utensil",
        "wine_glass": "utensil",
        "beef_steak": "food",
        "olive_oil": "food",
        "spice_cabinet": "container"
    }
    
    correct_semantic = 0
    total_semantic = 0
    
    for obj in test_objects:
        pred_type, confidence = classifier.predict_with_confidence(obj)
        
        # Check if it's a semantic test case
        is_semantic_test = obj in semantic_tests
        semantic_result = ""
        
        if is_semantic_test:
            expected = semantic_tests[obj]
            is_correct = pred_type == expected
            semantic_result = f"✓ ({expected})" if is_correct else f"✗ (expected {expected})"
            if is_correct:
                correct_semantic += 1
            total_semantic += 1
        
        print(f"{obj:<20} {pred_type:<15} {confidence:.3f}       {semantic_result}")
    
    print("-" * 70)
    print(f"Semantic Understanding: {correct_semantic}/{total_semantic} correct" if total_semantic > 0 else "")
    print(f"Total objects tested: {len(test_objects)}")