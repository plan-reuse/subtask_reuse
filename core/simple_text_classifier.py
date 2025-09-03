"""Simple text classifier for object type inference."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple


class SimpleTextClassifier:
    """Simple TF-IDF + Logistic Regression for object classification."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=500)
        self.classifier = LogisticRegression(random_state=42)
        self.is_trained = False
    
    def train(self, training_data: List[Tuple[str, str]]):
        """Train with list of (object_name, object_type) tuples."""
        objects, types = zip(*training_data)
        
        X = self.vectorizer.fit_transform(objects)
        self.classifier.fit(X, types)
        self.is_trained = True
        
        print(f"Trained on {len(training_data)} examples")
    
    def predict(self, obj_name: str) -> str:
        """Predict object type."""
        if not self.is_trained:
            return obj_name
        
        X = self.vectorizer.transform([obj_name])
        return self.classifier.predict(X)[0]


def create_training_data() -> List[Tuple[str, str]]:
    """Create balanced training data from PDDL + synthetic examples."""
    return [
        # Regions (8 examples)
        ("kitchen_corner", "region"), ("countertop_1", "region"), ("countertop_2", "region"), ("sink", "region"),
        ("countertop", "region"), ("counter", "region"), ("kitchen_counter", "region"), ("countertop_main", "region"),
        
        # Containers (8 examples)  
        ("fridge", "container"), ("cabinet", "container"), ("refrigerator", "container"), ("cupboard", "container"), 
        ("drawer", "container"), ("fridge_main", "container"), ("cabinet_1", "container"), ("freezer", "container"),
        
        # Stovetops (8 examples)
        ("stove", "stovetop"), ("stovetop", "stovetop"), ("burner", "stovetop"), ("cooktop", "stovetop"),
        ("stove_1", "stovetop"), ("gas_stove", "stovetop"), ("electric_stove", "stovetop"), ("burner_1", "stovetop"),
        
        # Utensils (8 examples)
        ("plate", "utensil"), ("bowl", "utensil"), ("dish", "utensil"), ("spoon", "utensil"), 
        ("fork", "utensil"), ("knife", "utensil"), ("plate_2", "utensil"), ("dinner_plate", "utensil"),
        
        # Cookware (8 examples)
        ("pot", "cookware"), ("pan", "cookware"), ("saucepan", "cookware"), ("frying_pan", "cookware"), 
        ("skillet", "cookware"), ("pot_1", "cookware"), ("sauce_pan", "cookware"), ("cooking_pot", "cookware"),
        
        # Food (8 examples)
        ("salt", "food"), ("pepper", "food"), ("chicken", "food"), ("potato", "food"), 
        ("tomato", "food"), ("beef", "food"), ("cheese", "food"), ("chicken_1", "food"),
    ]


if __name__ == "__main__":
    print("=== Simple Text Classifier Test ===\n")
    
    # Create and train classifier
    classifier = SimpleTextClassifier()
    training_data = create_training_data()
    classifier.train(training_data)
    
    # Test objects: PDDL objects + novel objects
    test_objects = [
        # Original PDDL objects
        "pot", "plate", "chicken", "salt", "fridge", "stove", "countertop_1",
        
        # Novel/unseen objects
        "sauce_pan", "dinner_plate", "chicken_breast", "salt_shaker", 
        "refrigerator", "gas_stove", "kitchen_counter", "cooking_pot",
        "beef_steak", "cheese_slice", "bread_loaf", "egg_carton",
        "microwave", "toaster", "blender", "cutting_board",
        "wine_glass", "coffee_mug", "soup_bowl", "pasta_sauce",
        
        # Numbered variations
        "pot_1", "pot_main", "chicken_2", "plate_dining", "fridge_side",
        
        # Completely unknown
        "mystery_object", "thing_123", "unknown_appliance"
    ]
    
    print("Classification Results:")
    print("-" * 50)
    print(f"{'Object':<20} {'Predicted Type':<15} {'Known/Novel'}")
    print("-" * 50)
    
    # Track known vs novel
    known_objects = [obj for obj, _ in training_data]
    
    for obj in test_objects:
        pred_type = classifier.predict(obj)
        status = "Known" if obj in known_objects else "Novel"
        print(f"{obj:<20} {pred_type:<15} {status}")
    
    print(f"\nTested {len(test_objects)} objects total")
    print(f"Known: {sum(1 for obj in test_objects if obj in known_objects)}")
    print(f"Novel: {sum(1 for obj in test_objects if obj not in known_objects)}")