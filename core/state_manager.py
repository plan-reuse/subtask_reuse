"""State management module for tracking world state changes."""

from typing import List, Set

from utils.parsing import extract_predicates_from_init, extract_original_init_state
from utils.effects import parse_action_predicates


class StateManager:
    """Manages world state updates based on action execution."""
    
    def __init__(self, initial_state: str = None):
        self.current_state = initial_state
    
    def update_state(self, action_plan: List[str], initial_state: str = None) -> str:
        """Update the initial state based on the executed action plan."""
        if initial_state:
            current_init_state = initial_state
        elif self.current_state:
            current_init_state = self.current_state
        else:
            raise ValueError("No initial state provided")
        
        print(f"\n--- Updating initial state based on {len(action_plan)} actions ---")
        
        # Extract predicates from current init state
        init_predicates = extract_predicates_from_init(current_init_state)
        
        print(f"Starting with {len(init_predicates)} initial predicates")
        
        # Apply effects of each action
        for i, action in enumerate(action_plan):
            print(f"Applying action {i+1}: {action}")
            effects = parse_action_predicates(action)
            print(f"  Generated effects: {effects}")
            
            for effect in effects:
                if effect.startswith("(not "):
                    # Remove predicate
                    predicate_to_remove = effect[5:-1]  # Remove "(not " and ")"
                    # Handle different robot name variations
                    predicates_to_check = [predicate_to_remove]
                    if "pr2" in predicate_to_remove:
                        # Also check for variations with different robot names
                        alt_predicate = predicate_to_remove.replace("pr2", "stretch_robot")
                        predicates_to_check.append(alt_predicate)
                    elif "stretch_robot" in predicate_to_remove:
                        # Also check for pr2 robot name
                        alt_predicate = predicate_to_remove.replace("stretch_robot", "pr2")
                        predicates_to_check.append(alt_predicate)
                    
                    removed = False
                    for pred_check in predicates_to_check:
                        if pred_check in init_predicates:
                            init_predicates.remove(pred_check)
                            print(f"  Removed: {pred_check}")
                            removed = True
                            break
                    
                    if not removed:
                        print(f"  Could not remove (not found): {predicate_to_remove}")
                else:
                    # Add predicate
                    init_predicates.add(effect)
                    print(f"  Added: {effect}")
        
        print(f"Final state has {len(init_predicates)} predicates")
        
        # Reconstruct the init state string
        updated_init_state = "(:init\n"
        for predicate in sorted(init_predicates):
            updated_init_state += f"    {predicate}\n"
        updated_init_state += "  )"
        
        # Update current state
        self.current_state = updated_init_state
        
        return updated_init_state
    
    def get_current_state(self) -> str:
        """Get the current state."""
        return self.current_state
    
    def reset_state(self, initial_state: str) -> None:
        """Reset to a specific initial state."""
        self.current_state = initial_state
    
    def load_original_state(self, problem_file: str) -> str:
        """Load original state from problem file."""
        original_init = extract_original_init_state(problem_file)
        self.current_state = original_init
        return original_init