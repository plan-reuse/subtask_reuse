"""Plan abstraction and instantiation module."""

from typing import List, Dict, Tuple

from utils.parsing import extract_robot_location


class PlanAbstractor:
    """Handles plan abstraction and instantiation."""
    
    def __init__(self):
        # Define object type mappings
        self.object_types = {
            'cookware': ['pot'],
            'utensil': ['plate'],
            "food": ["salt", "pepper", "chicken", "potato", "tomato", "cabbage", "zucchini", "artichoke", "oil", "vinegar", "milk"],
            'container': ['fridge', 'cabinet']
        }
    
    def get_object_type(self, obj_name: str) -> str:
        """Get the type of an object based on predefined object lists."""
        for obj_type, objects in self.object_types.items():
            if obj_name in objects:
                return obj_type
        return obj_name  # Return as-is if not in abstraction list
    
    def abstract_predicate(self, predicate: str) -> str:
        """Abstract a PDDL predicate by replacing object names with their types."""
        # Split predicate into parts
        predicate = predicate.strip('()')
        parts = predicate.split()
        
        if not parts:
            return predicate
        
        predicate_name = parts[0]
        
        # Abstract based on predicate type
        if predicate_name in ['at', 'inside', 'in_cookware', 'on_utensil', 'holding']:
            if len(parts) >= 3:
                obj = parts[1]
                location = parts[2]
                obj_type = self.get_object_type(obj)
                location_type = self.get_object_type(location)
                
                # Abstract if we have a type mapping
                if obj_type != obj:
                    obj = f"<{obj_type}>"
                if location_type != location:
                    location = f"<{location_type}>"
                    
                return f"({predicate_name} {obj} {location})"
        
        elif predicate_name in ['door_open', 'door_closed', 'turned_on', 'turned_off']:
            if len(parts) >= 2:
                obj = parts[1]
                obj_type = self.get_object_type(obj)
                if obj_type != obj:
                    obj = f"<{obj_type}>"
                return f"({predicate_name} {obj})"
        
        elif predicate_name == 'on_stovetop':
            if len(parts) >= 3:
                cookware = parts[1]
                stovetop = parts[2]
                cookware_type = self.get_object_type(cookware)
                if cookware_type != cookware:
                    cookware = f"<{cookware_type}>"
                return f"({predicate_name} {cookware} {stovetop})"
        
        elif predicate_name == 'cooked':
            if len(parts) >= 2:
                food = parts[1]
                food_type = self.get_object_type(food)
                if food_type != food:
                    food = f"<{food_type}>"
                return f"({predicate_name} {food})"
        
        # Return original predicate if no abstraction applied
        return f"({predicate})"
    
    def abstract_goal(self, pddl_goal: str) -> str:
        """Abstract a PDDL goal by replacing object names with their types."""
        # Handle compound goals with 'and'
        if pddl_goal.strip().startswith("(and"):
            # Extract all predicates from the compound goal
            predicates = []
            temp_goal = pddl_goal.strip()
            
            # Remove the outer "(and" and the final ")"
            if temp_goal.startswith("(and "):
                temp_goal = temp_goal[5:]  # Remove "(and "
            if temp_goal.endswith(")"):
                temp_goal = temp_goal[:-1]  # Remove final ")"
            
            # Extract individual predicates using balanced parentheses
            i = 0
            while i < len(temp_goal):
                if temp_goal[i] == '(':
                    # Found start of predicate, find matching closing parenthesis
                    paren_count = 1
                    start = i
                    i += 1
                    while i < len(temp_goal) and paren_count > 0:
                        if temp_goal[i] == '(':
                            paren_count += 1
                        elif temp_goal[i] == ')':
                            paren_count -= 1
                        i += 1
                    
                    if paren_count == 0:
                        predicate = temp_goal[start:i].strip()
                        if predicate and not predicate.startswith('(and'):
                            predicates.append(predicate)
                else:
                    i += 1
            
            # Abstract each predicate
            abstracted_predicates = []
            for pred in predicates:
                abstracted_pred = self.abstract_predicate(pred)
                abstracted_predicates.append(abstracted_pred)
            
            # Return single predicate or compound goal
            if len(abstracted_predicates) == 1:
                return abstracted_predicates[0]
            else:
                return "(and " + " ".join(abstracted_predicates) + ")"
        else:
            # Single predicate goal
            return self.abstract_predicate(pddl_goal)
    
    def abstract_action(self, action: str) -> str:
        """Abstract a single action by replacing object names with their types."""
        parts = action.split()
        if not parts:
            return action
        
        action_name = parts[0]
        
        # Abstract different action types
        if action_name == "move":
            # move pr2 from_loc to_loc
            if len(parts) >= 4:
                robot, from_loc, to_loc = parts[1], parts[2], parts[3]
                from_type = self.get_object_type(from_loc)
                to_type = self.get_object_type(to_loc)
                
                if from_type != from_loc:
                    from_loc = f"<{from_type}>"
                if to_type != to_loc:
                    to_loc = f"<{to_type}>"
                    
                return f"{action_name} {robot} {from_loc} {to_loc}"
        
        elif action_name in ["open_container", "close_container"]:
            # open_container pr2 container
            if len(parts) >= 3:
                robot, container = parts[1], parts[2]
                container_type = self.get_object_type(container)
                if container_type != container:
                    container = f"<{container_type}>"
                return f"{action_name} {robot} {container}"
        
        elif action_name in ["pickup_from_container", "putdown_to_container"]:
            # pickup_from_container pr2 obj container
            if len(parts) >= 4:
                robot, obj, container = parts[1], parts[2], parts[3]
                obj_type = self.get_object_type(obj)
                container_type = self.get_object_type(container)
                
                if obj_type != obj:
                    obj = f"<{obj_type}>"
                if container_type != container:
                    container = f"<{container_type}>"
                    
                return f"{action_name} {robot} {obj} {container}"
        
        elif action_name in ["pickup_from_region", "putdown_to_region"]:
            # pickup_from_region pr2 obj location
            if len(parts) >= 4:
                robot, obj, location = parts[1], parts[2], parts[3]
                obj_type = self.get_object_type(obj)
                location_type = self.get_object_type(location)
                
                if obj_type != obj:
                    obj = f"<{obj_type}>"
                if location_type != location:
                    location = f"<{location_type}>"
                    
                return f"{action_name} {robot} {obj} {location}"
        
        elif action_name == "put_food_in_cookware":
            # put_food_in_cookware pr2 food cookware location
            if len(parts) >= 5:
                robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
                food_type = self.get_object_type(food)
                cookware_type = self.get_object_type(cookware)
                location_type = self.get_object_type(location)
                
                if food_type != food:
                    food = f"<{food_type}>"
                if cookware_type != cookware:
                    cookware = f"<{cookware_type}>"
                if location_type != location:
                    location = f"<{location_type}>"
                    
                return f"{action_name} {robot} {food} {cookware} {location}"
        
        elif action_name == "place_cookware_on_stovetop":
            # place_cookware_on_stovetop pr2 cookware stovetop
            if len(parts) >= 4:
                robot, cookware, stovetop = parts[1], parts[2], parts[3]
                cookware_type = self.get_object_type(cookware)
                if cookware_type != cookware:
                    cookware = f"<{cookware_type}>"
                return f"{action_name} {robot} {cookware} {stovetop}"
        
        elif action_name == "remove_cookware_from_stovetop":
            # remove_cookware_from_stovetop pr2 cookware stovetop
            if len(parts) >= 4:
                robot, cookware, stovetop = parts[1], parts[2], parts[3]
                cookware_type = self.get_object_type(cookware)
                if cookware_type != cookware:
                    cookware = f"<{cookware_type}>"
                return f"{action_name} {robot} {cookware} {stovetop}"
        
        elif action_name in ["turn_on_stove", "turn_off_stove"]:
            # turn_on_stove pr2 stove
            return action  # No abstraction needed for stove actions
        
        elif action_name == "wait_for_food_to_cook":
            # wait_for_food_to_cook stovetop cookware food
            if len(parts) >= 4:
                stovetop, cookware, food = parts[1], parts[2], parts[3]
                cookware_type = self.get_object_type(cookware)
                food_type = self.get_object_type(food)
                
                if cookware_type != cookware:
                    cookware = f"<{cookware_type}>"
                if food_type != food:
                    food = f"<{food_type}>"
                    
                return f"{action_name} {stovetop} {cookware} {food}"
        
        # Return original action if no abstraction applied
        return action
    
    def create_search_key(self, pddl_goal: str, robot_init_location: str) -> Tuple[str, str]:
        """Create a search key using abstracted goal and robot initial location."""
        abstract_goal_str = self.abstract_goal(pddl_goal)
        robot_init_str = robot_init_location if robot_init_location else "Unknown"
        
        return (f"goal: {abstract_goal_str}", f"robot_init: {robot_init_str}")
    
    def instantiate_abstract_plan(self, abstract_plan: List[str], pddl_goal: str, robot_init_location: str) -> List[str]:
        """Instantiate an abstract plan with concrete objects from the goal and context."""
        
        # Extract objects from the goal
        goal_objects = {}
        
        # Parse goal to extract object mappings
        if pddl_goal.strip().startswith("(and"):
            # Extract predicates from compound goal
            temp_goal = pddl_goal.strip()
            if temp_goal.startswith("(and "):
                temp_goal = temp_goal[5:]
            if temp_goal.endswith(")"):
                temp_goal = temp_goal[:-1]
            
            # Extract individual predicates
            predicates = []
            i = 0
            while i < len(temp_goal):
                if temp_goal[i] == '(':
                    paren_count = 1
                    start = i
                    i += 1
                    while i < len(temp_goal) and paren_count > 0:
                        if temp_goal[i] == '(':
                            paren_count += 1
                        elif temp_goal[i] == ')':
                            paren_count -= 1
                        i += 1
                    
                    if paren_count == 0:
                        predicate = temp_goal[start:i].strip()
                        if predicate:
                            predicates.append(predicate)
                else:
                    i += 1
            
            # Extract objects from each predicate
            for pred in predicates:
                self._extract_objects_from_predicate(pred, goal_objects)
        else:
            # Single predicate goal
            self._extract_objects_from_predicate(pddl_goal, goal_objects)
        
        # Extract robot location info and current location
        if robot_init_location:
            robot_parts = robot_init_location.strip('()').split()
            if len(robot_parts) >= 3:
                goal_objects['robot'] = robot_parts[1]  # pr2
                current_location = robot_parts[2]
                goal_objects['current_location'] = current_location
                
                # Only map <container> to current location if it's not already defined from goal
                # This handles movement actions where robot moves FROM current location
                if '<container>' not in goal_objects:
                    goal_objects['<container>'] = current_location
        
        print(f"Extracted objects for instantiation: {goal_objects}")
        
        # Instantiate the abstract plan
        instantiated_plan = []
        for abstract_action in abstract_plan:
            instantiated_action = self._instantiate_action_context_aware(abstract_action, goal_objects)
            instantiated_plan.append(instantiated_action)
        
        return instantiated_plan
    
    def _extract_objects_from_predicate(self, predicate: str, goal_objects: Dict[str, str]) -> None:
        """Extract objects from a single predicate and add to goal_objects mapping."""
        predicate = predicate.strip('()')
        parts = predicate.split()
        
        if not parts:
            return
        
        predicate_name = parts[0]
        
        if predicate_name in ['at', 'inside', 'in_cookware', 'on_utensil']:
            if len(parts) >= 3:
                obj = parts[1]
                location = parts[2]
                obj_type = self.get_object_type(obj)
                location_type = self.get_object_type(location)
                
                if obj_type != obj:
                    goal_objects[f'<{obj_type}>'] = obj
                if location_type != location:
                    goal_objects[f'<{location_type}>'] = location
                else:
                    # If location is not abstracted, map it directly for specific locations
                    if location in ['countertop', 'stove', 'kitchen_corner']:
                        goal_objects[location] = location
        
        elif predicate_name == 'holding':
            if len(parts) >= 3:
                robot = parts[1]
                obj = parts[2]
                obj_type = self.get_object_type(obj)
                
                # Always map the robot
                goal_objects['robot'] = robot
                
                # Map the object being held
                if obj_type != obj:
                    goal_objects[f'<{obj_type}>'] = obj
                else:
                    goal_objects[obj] = obj
        
        elif predicate_name in ['door_closed', 'door_open', 'turned_on', 'turned_off']:
            if len(parts) >= 2:
                obj = parts[1]
                obj_type = self.get_object_type(obj)
                if obj_type != obj:
                    goal_objects[f'<{obj_type}>'] = obj
                else:
                    # Map specific objects directly
                    if obj in ['stove']:
                        goal_objects[obj] = obj
        
        elif predicate_name == 'on_stovetop':
            if len(parts) >= 3:
                cookware = parts[1]
                stovetop = parts[2]
                cookware_type = self.get_object_type(cookware)
                if cookware_type != cookware:
                    goal_objects[f'<{cookware_type}>'] = cookware
                # Always map stove/stovetop
                goal_objects['stove'] = stovetop
                goal_objects['stovetop'] = stovetop
    
    def _instantiate_action_context_aware(self, abstract_action: str, goal_objects: Dict[str, str]) -> str:
        """Instantiate a single abstract action with context-aware object replacement."""
        instantiated = abstract_action
        
        # Handle movement actions specifically
        if abstract_action.startswith("move"):
            parts = abstract_action.split()
            if len(parts) >= 4:
                # For "move robot <container> location", <container> should be current_location
                # For "move robot location <container>", <container> should be goal container
                from_loc = parts[2]
                to_loc = parts[3]
                
                if from_loc == '<container>' and 'current_location' in goal_objects:
                    # Replace the "from" <container> with current location
                    instantiated = instantiated.replace('<container>', goal_objects['current_location'], 1)
                
                # Replace remaining placeholders with goal objects
                for placeholder, concrete_obj in goal_objects.items():
                    if placeholder not in ['robot', 'current_location']:
                        instantiated = instantiated.replace(placeholder, concrete_obj)
        else:
            # For non-movement actions, replace all placeholders normally
            for placeholder, concrete_obj in goal_objects.items():
                if placeholder not in ['robot', 'current_location']:
                    instantiated = instantiated.replace(placeholder, concrete_obj)
        
        return instantiated