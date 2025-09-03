import os
import time
import subprocess
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_output_directory():
    """Create a timestamped output directory and return its path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = "/home/shaid/Documents/PDDL/output"
    output_dir = os.path.join(output_base, f"run_{timestamp}")
    subtask_problem_dir = os.path.join(output_dir, "subtask_problem")
    
    # Create the directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(subtask_problem_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    print(f"Created subtask problem directory: {subtask_problem_dir}")
    
    return output_dir, subtask_problem_dir

def load_knowledge_base():
    """Load the existing knowledge base from file"""
    kb_file = "/home/shaid/Documents/PDDL/output/knowledge_base.txt"
    
    if not os.path.exists(kb_file):
        print("No existing knowledge base found. Starting with empty knowledge base.")
        return {}
    
    # Read and execute the knowledge base file
    with open(kb_file, 'r') as f:
        content = f.read()
    
    # Extract the knowledge_base dictionary from the file
    local_vars = {}
    exec(content, {}, local_vars)
    knowledge_base = local_vars.get('knowledge_base', {})
    
    print(f"Loaded knowledge base with {len(knowledge_base)} entries")
    return knowledge_base

def save_knowledge_base(knowledge_base, output_dir):
    """Save knowledge base to a file in dictionary format"""
    kb_file = os.path.join(output_dir, "knowledge_base.txt")
    
    with open(kb_file, 'w') as f:
        f.write("knowledge_base = {\n")
        
        for i, (key, actions) in enumerate(knowledge_base.items()):
            goal_str, robot_init_str = key
            
            f.write("    (\n")
            f.write(f'        "{goal_str}",\n')
            f.write(f'        "{robot_init_str}"\n')
            f.write("    ): [\n")
            
            for action in actions:
                f.write(f'        "{action}",\n')
            
            f.write("    ]")
            
            # Add comma if not the last entry
            if i < len(knowledge_base) - 1:
                f.write(",")
            f.write("\n")
        
        f.write("}\n")
    
    print(f"Knowledge base saved to: {kb_file}")
    return kb_file

def read_subtasks_file(subtasks_file_path):
    """Read and parse the manually created subtasks file"""
    if not os.path.exists(subtasks_file_path):
        raise FileNotFoundError(f"Subtasks file not found: {subtasks_file_path}")
    
    with open(subtasks_file_path, 'r') as f:
        content = f.read()
    
    print(f"Successfully read subtasks file: {subtasks_file_path}")
    return content

def parse_subtasks_and_goals(plan_text):
    """Parse the subtasks file to extract subtasks and PDDL goals"""
    lines = plan_text.strip().split('\n')
    subtasks = []
    pddl_goals = []
    
    current_subtask = ""
    current_goals = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Subtask"):
            # Save previous subtask if exists
            if current_subtask and current_goals:
                subtasks.append(current_subtask)
                # Join multiple goals with 'and' if there are multiple
                if len(current_goals) == 1:
                    pddl_goals.append(current_goals[0])
                else:
                    combined_goal = "(and " + " ".join(current_goals) + ")"
                    pddl_goals.append(combined_goal)
            
            # Start new subtask
            current_subtask = line.split(":", 1)[1].strip() if ":" in line else ""
            current_goals = []
            
        elif line.startswith("PDDL Goal"):
            goal = line.split(":", 1)[1].strip() if ":" in line else ""
            if goal:
                current_goals.append(goal)
    
    # Add the last subtask
    if current_subtask and current_goals:
        subtasks.append(current_subtask)
        if len(current_goals) == 1:
            pddl_goals.append(current_goals[0])
        else:
            combined_goal = "(and " + " ".join(current_goals) + ")"
            pddl_goals.append(combined_goal)
    
    return subtasks, pddl_goals

def get_object_type(obj_name):
    """Get the type of an object based on predefined object lists"""
    # Define object types based on the domain
    cookware_objects = ['pot', 'pan']
    utensil_objects = ['bowl', 'plate'] 
    food_objects = ['pancake', 'bread', 'salt', 'butter', 'chicken', 'fish', 'potato', 'tomato', 'cabbage', 'lettuce', 'apple']
    container_objects = ['fridge', 'cabinet']
    
    if obj_name in cookware_objects:
        return 'cookware'
    elif obj_name in utensil_objects:
        return 'utensil'
    elif obj_name in food_objects:
        return 'food'
    elif obj_name in container_objects:
        return 'container'
    else:
        return obj_name  # Return as-is if not in abstraction list

def abstract_predicate(predicate):
    """Abstract a PDDL predicate by replacing object names with their types"""
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
            obj_type = get_object_type(obj)
            location_type = get_object_type(location)
            
            # Abstract if we have a type mapping
            if obj_type != obj:
                obj = f"<{obj_type}>"
            if location_type != location:
                location = f"<{location_type}>"
                
            return f"({predicate_name} {obj} {location})"
    
    elif predicate_name in ['door_open', 'door_closed', 'turned_on', 'turned_off']:
        if len(parts) >= 2:
            obj = parts[1]
            obj_type = get_object_type(obj)
            if obj_type != obj:
                obj = f"<{obj_type}>"
            return f"({predicate_name} {obj})"
    
    elif predicate_name == 'on_stovetop':
        if len(parts) >= 3:
            cookware = parts[1]
            stovetop = parts[2]
            cookware_type = get_object_type(cookware)
            if cookware_type != cookware:
                cookware = f"<{cookware_type}>"
            return f"({predicate_name} {cookware} {stovetop})"
    
    elif predicate_name == 'cooked':
        if len(parts) >= 2:
            food = parts[1]
            food_type = get_object_type(food)
            if food_type != food:
                food = f"<{food_type}>"
            return f"({predicate_name} {food})"
    
    # Return original predicate if no abstraction applied
    return f"({predicate})"

def abstract_goal(pddl_goal):
    """Abstract a PDDL goal by replacing object names with their types"""
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
            abstracted_pred = abstract_predicate(pred)
            abstracted_predicates.append(abstracted_pred)
        
        # Return single predicate or compound goal
        if len(abstracted_predicates) == 1:
            return abstracted_predicates[0]
        else:
            return "(and " + " ".join(abstracted_predicates) + ")"
    else:
        # Single predicate goal
        return abstract_predicate(pddl_goal)

def extract_robot_location(init_state_text, robot_name="stretch_robot"):
    """Extract the robot's location from the initial state"""
    predicates = extract_predicates_from_init(init_state_text)
    
    for predicate in predicates:
        # Look for any robot location predicate, not just with the specific robot name
        if predicate.startswith("(at_robot"):
            parts = predicate.strip('()').split()
            if len(parts) >= 3:
                # Extract the actual robot name and location
                actual_robot = parts[1]
                location = parts[2]
                # Return in standardized format using "stretch_robot"
                return f"(at_robot stretch_robot {location})"
    
    return None

def extract_predicates_from_init(init_state_text):
    """Extract all predicates from an init state text"""
    predicates = set()
    
    # Remove the (:init and closing )
    content = init_state_text.strip()
    if content.startswith("(:init"):
        content = content[6:]  # Remove "(:init"
    if content.endswith(")"):
        content = content[:-1]  # Remove closing ")"
    
    # Split into lines and process each
    lines = content.split('\n')
    current_predicate = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle multi-line predicates
        current_predicate += " " + line if current_predicate else line
        
        # Count parentheses to determine if predicate is complete
        open_count = current_predicate.count('(')
        close_count = current_predicate.count(')')
        
        if open_count == close_count and open_count > 0:
            # Complete predicate found
            pred = current_predicate.strip()
            if pred and not pred.startswith('(:'):
                predicates.add(pred)
            current_predicate = ""
    
    return predicates

def create_search_key(pddl_goal, robot_init_location):
    """Create a search key using abstracted goal and robot initial location"""
    abstract_goal_str = abstract_goal(pddl_goal)
    robot_init_str = robot_init_location if robot_init_location else "Unknown"
    
    return (f"goal: {abstract_goal_str}", f"robot_init: {robot_init_str}")

def instantiate_abstract_plan(abstract_plan, pddl_goal, robot_init_location):
    """Instantiate an abstract plan with concrete objects from the goal and context"""
    
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
            extract_objects_from_predicate(pred, goal_objects)
    else:
        # Single predicate goal
        extract_objects_from_predicate(pddl_goal, goal_objects)
    
    # Extract robot location info and current location
    if robot_init_location:
        robot_parts = robot_init_location.strip('()').split()
        if len(robot_parts) >= 3:
            goal_objects['robot'] = robot_parts[1]  # stretch_robot
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
        instantiated_action = instantiate_action_context_aware(abstract_action, goal_objects)
        instantiated_plan.append(instantiated_action)
    
    return instantiated_plan

def instantiate_action_context_aware(abstract_action, goal_objects):
    """Instantiate a single abstract action with context-aware object replacement"""
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

def extract_objects_from_predicate(predicate, goal_objects):
    """Extract objects from a single predicate and add to goal_objects mapping"""
    predicate = predicate.strip('()')
    parts = predicate.split()
    
    if not parts:
        return
    
    predicate_name = parts[0]
    
    if predicate_name in ['at', 'inside', 'in_cookware', 'on_utensil']:
        if len(parts) >= 3:
            obj = parts[1]
            location = parts[2]
            obj_type = get_object_type(obj)
            location_type = get_object_type(location)
            
            if obj_type != obj:
                goal_objects[f'<{obj_type}>'] = obj
            if location_type != location:
                goal_objects[f'<{location_type}>'] = location
            else:
                # If location is not abstracted, map it directly for specific locations
                if location in ['countertop', 'stove', 'kitchen_corner']:
                    goal_objects[location] = location
    
    elif predicate_name in ['door_closed', 'door_open', 'turned_on', 'turned_off']:
        if len(parts) >= 2:
            obj = parts[1]
            obj_type = get_object_type(obj)
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
            cookware_type = get_object_type(cookware)
            if cookware_type != cookware:
                goal_objects[f'<{cookware_type}>'] = cookware
            # Always map stove/stovetop
            goal_objects['stove'] = stovetop
            goal_objects['stovetop'] = stovetop



def create_problem_file_with_goal(empty_problem_file, output_problem_file, pddl_goal):
    """Create a new problem file with the specified PDDL goal"""
    with open(empty_problem_file, 'r') as f:
        content = f.read()
    
    # Replace the empty goal section with the actual goal
    goal_section = f"(:goal\n    {pddl_goal}\n  )"
    updated_content = re.sub(r'\(:goal\s*\n\s*\)', goal_section, content)
    
    with open(output_problem_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Created problem file with goal: {output_problem_file}")

def run_classical_planner(domain_file, problem_file, search_option='astar(blind())'):
    """Run the Fast Downward classical planner and extract the action plan"""
    
    # Configuration
    PDDL_ROOT = "/home/shaid/Documents/PDDL"
    FAST_DOWNWARD_DIR = os.path.join(PDDL_ROOT, "downward")
    fd_script = os.path.join(FAST_DOWNWARD_DIR, "fast-downward.py")
    
    command = [
        "python3", fd_script,
        domain_file,
        problem_file,
        "--search", search_option
    ]
    
    print(f"Running classical planner with command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        plan_lines = []
        stdout_lines = result.stdout.split('\n')
        
        plan_started = False
        for line in stdout_lines:
            if line.startswith('[') or line.startswith('INFO') or not line.strip():
                if plan_started and line.startswith('['):
                    plan_started = False
                continue
            
            if ('(' in line and ')' in line) or plan_started:
                plan_started = True
                clean_line = re.sub(r'\s+\(\d+\)$', '', line)
                plan_lines.append(clean_line)
        
        if plan_lines:
            return plan_lines, True
        else:
            if "Solution found" in result.stdout:
                return ["A plan was found but couldn't be extracted properly."], False
            else:
                return ["No plan found."], False
                
    except subprocess.TimeoutExpired:
        return ["Planner timed out after 60 seconds."], False
    except Exception as e:
        return [f"Error running planner: {str(e)}"], False

def abstract_action(action):
    """Abstract a single action by replacing object names with their types"""
    parts = action.split()
    if not parts:
        return action
    
    action_name = parts[0]
    
    # Abstract different action types
    if action_name == "move":
        # move stretch_robot from_loc to_loc
        if len(parts) >= 4:
            robot, from_loc, to_loc = parts[1], parts[2], parts[3]
            from_type = get_object_type(from_loc)
            to_type = get_object_type(to_loc)
            
            if from_type != from_loc:
                from_loc = f"<{from_type}>"
            if to_type != to_loc:
                to_loc = f"<{to_type}>"
                
            return f"{action_name} {robot} {from_loc} {to_loc}"
    
    elif action_name in ["open_container", "close_container"]:
        # open_container stretch_robot container
        if len(parts) >= 3:
            robot, container = parts[1], parts[2]
            container_type = get_object_type(container)
            if container_type != container:
                container = f"<{container_type}>"
            return f"{action_name} {robot} {container}"
    
    elif action_name in ["pickup_from_container", "putdown_to_container"]:
        # pickup_from_container stretch_robot obj container
        if len(parts) >= 4:
            robot, obj, container = parts[1], parts[2], parts[3]
            obj_type = get_object_type(obj)
            container_type = get_object_type(container)
            
            if obj_type != obj:
                obj = f"<{obj_type}>"
            if container_type != container:
                container = f"<{container_type}>"
                
            return f"{action_name} {robot} {obj} {container}"
    
    elif action_name in ["pickup_from_region", "putdown_to_region"]:
        # pickup_from_region stretch_robot obj location
        if len(parts) >= 4:
            robot, obj, location = parts[1], parts[2], parts[3]
            obj_type = get_object_type(obj)
            location_type = get_object_type(location)
            
            if obj_type != obj:
                obj = f"<{obj_type}>"
            if location_type != location:
                location = f"<{location_type}>"
                
            return f"{action_name} {robot} {obj} {location}"
    
    elif action_name == "put_food_in_cookware":
        # put_food_in_cookware stretch_robot food cookware location
        if len(parts) >= 5:
            robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
            food_type = get_object_type(food)
            cookware_type = get_object_type(cookware)
            location_type = get_object_type(location)
            
            if food_type != food:
                food = f"<{food_type}>"
            if cookware_type != cookware:
                cookware = f"<{cookware_type}>"
            if location_type != location:
                location = f"<{location_type}>"
                
            return f"{action_name} {robot} {food} {cookware} {location}"
    
    elif action_name == "place_cookware_on_stovetop":
        # place_cookware_on_stovetop stretch_robot cookware stovetop
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            cookware_type = get_object_type(cookware)
            if cookware_type != cookware:
                cookware = f"<{cookware_type}>"
            return f"{action_name} {robot} {cookware} {stovetop}"
    
    elif action_name == "remove_cookware_from_stovetop":
        # remove_cookware_from_stovetop stretch_robot cookware stovetop
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            cookware_type = get_object_type(cookware)
            if cookware_type != cookware:
                cookware = f"<{cookware_type}>"
            return f"{action_name} {robot} {cookware} {stovetop}"
    
    elif action_name in ["turn_on_stove", "turn_off_stove"]:
        # turn_on_stove stretch_robot stove
        return action  # No abstraction needed for stove actions
    
    elif action_name == "wait_for_food_to_cook":
        # wait_for_food_to_cook stovetop cookware food
        if len(parts) >= 4:
            stovetop, cookware, food = parts[1], parts[2], parts[3]
            cookware_type = get_object_type(cookware)
            food_type = get_object_type(food)
            
            if cookware_type != cookware:
                cookware = f"<{cookware_type}>"
            if food_type != food:
                food = f"<{food_type}>"
                
            return f"{action_name} {stovetop} {cookware} {food}"
    
    # Return original action if no abstraction applied
    return action

def parse_action_predicates(action):
    """Parse a PDDL action to extract the predicates it affects based on the domain"""
    effects = []
    
    # Clean the action string
    action = action.strip()
    
    # Movement effects (domain uses at_robot predicate)
    if action.startswith("move "):
        parts = action.split()
        if len(parts) >= 4:
            robot, from_loc, to_loc = parts[1], parts[2], parts[3]
            # Normalize robot name for consistency
            normalized_robot = "stretch_robot"
            effects.append(f"(not (at_robot {normalized_robot} {from_loc}))")
            effects.append(f"(at_robot {normalized_robot} {to_loc})")
    
    # Container opening effects
    elif action.startswith("open_container "):
        parts = action.split()
        if len(parts) >= 3:
            robot, container = parts[1], parts[2]
            effects.append(f"(door_open {container})")
            effects.append(f"(not (door_closed {container}))")
    
    # Container closing effects
    elif action.startswith("close_container "):
        parts = action.split()
        if len(parts) >= 3:
            robot, container = parts[1], parts[2]
            effects.append(f"(door_closed {container})")
            effects.append(f"(not (door_open {container}))")
    
    # Pickup from region
    elif action.startswith("pickup_from_region "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, location = parts[1], parts[2], parts[3]
            normalized_robot = "stretch_robot"
            effects.append(f"(holding {normalized_robot} {obj})")
            effects.append(f"(not (at {obj} {location}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Putdown to region
    elif action.startswith("putdown_to_region "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, location = parts[1], parts[2], parts[3]
            normalized_robot = "stretch_robot"
            effects.append(f"(at {obj} {location})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {obj}))")
    
    # Pickup from container
    elif action.startswith("pickup_from_container "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, container = parts[1], parts[2], parts[3]
            normalized_robot = "stretch_robot"
            effects.append(f"(holding {normalized_robot} {obj})")
            effects.append(f"(not (inside {obj} {container}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Putdown to container
    elif action.startswith("putdown_to_container "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, container = parts[1], parts[2], parts[3]
            normalized_robot = "stretch_robot"
            effects.append(f"(inside {obj} {container})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {obj}))")
    
    # Put food in cookware
    elif action.startswith("put_food_in_cookware "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "stretch_robot"
            effects.append(f"(in_cookware {food} {cookware})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {food}))")
    
    # Take food from cookware
    elif action.startswith("take_food_from_cookware "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "stretch_robot"
            effects.append(f"(holding {normalized_robot} {food})")
            effects.append(f"(not (in_cookware {food} {cookware}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Place food on utensil
    elif action.startswith("place_food_on_utensil "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, utensil, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "stretch_robot"
            effects.append(f"(on_utensil {food} {utensil})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {food}))")
    
    # Take food from utensil
    elif action.startswith("take_food_from_utensil "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, utensil, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "stretch_robot"
            effects.append(f"(holding {normalized_robot} {food})")
            effects.append(f"(not (on_utensil {food} {utensil}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Place cookware on stovetop - robot stays at stove location
    elif action.startswith("place_cookware_on_stovetop "):
        parts = action.split()
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            normalized_robot = "stretch_robot"
            effects.append(f"(on_stovetop {cookware} {stovetop})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {cookware}))")
            # Robot implicitly stays at the stove location after placing cookware
            # Note: This action assumes robot is already at stove from previous move action
    
    # Remove cookware from stovetop
    elif action.startswith("remove_cookware_from_stovetop "):
        parts = action.split()
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            normalized_robot = "stretch_robot"
            effects.append(f"(holding {normalized_robot} {cookware})")
            effects.append(f"(not (on_stovetop {cookware} {stovetop}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Turn on stove
    elif action.startswith("turn_on_stove "):
        parts = action.split()
        if len(parts) >= 3:
            robot, stove = parts[1], parts[2]
            effects.append(f"(turned_on {stove})")
            effects.append(f"(not (turned_off {stove}))")
    
    # Turn off stove
    elif action.startswith("turn_off_stove "):
        parts = action.split()
        if len(parts) >= 3:
            robot, stove = parts[1], parts[2]
            effects.append(f"(turned_off {stove})")
            effects.append(f"(not (turned_on {stove}))")
    
    # Wait for food to cook
    elif action.startswith("wait_for_food_to_cook "):
        parts = action.split()
        if len(parts) >= 4:
            stovetop, cookware, food = parts[1], parts[2], parts[3]
            effects.append(f"(cooked {food})")
    
    return effects

def update_initial_state(current_init_state, action_plan):
    """Update the initial state based on the executed action plan"""
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
                if "stretch_robot" in predicate_to_remove:
                    # Also check for variations with doubled robot name
                    alt_predicate = predicate_to_remove.replace("stretch_robot", "stretch_stretch_robot")
                    predicates_to_check.append(alt_predicate)
                elif "stretch_stretch_robot" in predicate_to_remove:
                    # Also check for normal robot name
                    alt_predicate = predicate_to_remove.replace("stretch_stretch_robot", "stretch_robot")
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
    
    return updated_init_state

def create_updated_problem_file(base_problem_file, updated_init_state, pddl_goal, output_file):
    """Create a new problem file with updated initial state and goal"""
    with open(base_problem_file, 'r') as f:
        content = f.read()
    
    # Find the current init section and replace it
    # Use more precise regex that handles multi-line init sections
    init_pattern = r'\(:init\s*\n.*?\n\s*\)'
    
    # Check if init section exists
    init_match = re.search(init_pattern, content, re.DOTALL)
    if init_match:
        # Replace the existing init section
        updated_content = re.sub(init_pattern, updated_init_state, content, flags=re.DOTALL)
    else:
        # If no init section found, try to find (:init) and replace
        updated_content = re.sub(r'\(:init\s*\)', updated_init_state, content, flags=re.DOTALL)
    
    # Replace the goal section
    goal_section = f"(:goal\n    {pddl_goal}\n  )"
    updated_content = re.sub(r'\(:goal\s*\n?\s*\)', goal_section, updated_content, flags=re.DOTALL)
    
    # Write the updated content
    with open(output_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Created updated problem file: {output_file}")

def solve_subtask_with_kb(subtask_num, subtask, pddl_goal, domain_file, problem_file, 
                         subtask_problem_dir, knowledge_base, current_init_state=None):
    """Solve a single subtask using knowledge base retrieval or classical planning"""
    print(f"\n{'='*50}")
    print(f"SOLVING SUBTASK {subtask_num}")
    print(f"{'='*50}")
    print(f"Subtask: {subtask}")
    print(f"PDDL Goal: {pddl_goal}")
    
    # Extract robot's initial location for this subtask
    if current_init_state:
        robot_init_location = extract_robot_location(current_init_state)
    else:
        # For first subtask, read the original init state from the empty problem file
        with open(problem_file, 'r') as f:
            original_content = f.read()
        init_match = re.search(r'\(:init\s*\n(.*?)\n\s*\)', original_content, re.DOTALL)
        if init_match:
            init_content = init_match.group(1).strip()
            original_init = f"(:init\n{init_content}\n  )"
            robot_init_location = extract_robot_location(original_init)
        else:
            robot_init_location = None
    
    print(f"Robot initial location: {robot_init_location}")
    
    # Step 2: Create search key using abstracted goal and robot initial location
    search_key = create_search_key(pddl_goal, robot_init_location)
    print(f"Search key: {search_key}")
    
    # Debug: Print available knowledge base keys
    print(f"\nAvailable knowledge base keys:")
    for i, kb_key in enumerate(knowledge_base.keys(), 1):
        print(f"  {i}. {kb_key}")
    
    # Step 3: Check knowledge base for similar key
    if search_key in knowledge_base:
        print("🎯 Found matching entry in knowledge base!")
        abstract_plan = knowledge_base[search_key]
        print(f"Retrieved abstract plan:")
        for i, action in enumerate(abstract_plan, 1):
            print(f"  {i}. {action}")
        
        # Instantiate the abstract plan
        print("\n--- Instantiating abstract plan ---")
        action_plan = instantiate_abstract_plan(abstract_plan, pddl_goal, robot_init_location)
        print(f"Instantiated plan:")
        for i, action in enumerate(action_plan, 1):
            print(f"  {i}. {action}")
        
        success = True
        plan_source = "Knowledge Base"
    else:
        print("❌ No matching entry found in knowledge base. Using classical planner.")
        
        # Step 4: Use classical planner if no match found
        if current_init_state:
            # Use updated initial state
            temp_problem_file = os.path.join(subtask_problem_dir, f"temp_problem_subtask_{subtask_num}.pddl")
            create_updated_problem_file(problem_file, current_init_state, pddl_goal, temp_problem_file)
            problem_file_to_use = temp_problem_file
        else:
            # Use original problem file with goal
            problem_file_to_use = os.path.join(subtask_problem_dir, f"temp_problem_subtask_{subtask_num}.pddl")
            create_problem_file_with_goal(problem_file, problem_file_to_use, pddl_goal)
        
        # Run planner
        action_plan, success = run_classical_planner(domain_file, problem_file_to_use)
        plan_source = "Classical Planner"
        
        if success:
            # Abstract the plan and add to knowledge base
            print("\n--- Adding new entry to knowledge base ---")
            abstract_plan = []
            for action in action_plan:
                abstract_action_str = abstract_action(action)
                abstract_plan.append(abstract_action_str)
            
            knowledge_base[search_key] = abstract_plan
            print(f"Added new knowledge base entry:")
            print(f"Key: {search_key}")
            print(f"Abstract plan: {abstract_plan}")
    
    print(f"\n--- Action Plan for Subtask {subtask_num} (from {plan_source}) ---")
    if success:
        print(f"Successfully generated action plan with {len(action_plan)} steps:")
        for i, action in enumerate(action_plan, 1):
            print(f"{i}. {action}")
        
        # Update initial state if we have one
        if current_init_state:
            updated_init_state = update_initial_state(current_init_state, action_plan)
        else:
            # For first subtask, read the original init state from the empty problem file
            with open(problem_file, 'r') as f:
                original_content = f.read()
            # Extract original init state with better parsing
            init_match = re.search(r'\(:init\s*\n(.*?)\n\s*\)', original_content, re.DOTALL)
            if init_match:
                # Reconstruct the full init state
                init_content = init_match.group(1).strip()
                original_init = f"(:init\n{init_content}\n  )"
                print(f"\nOriginal init state extracted:")
                print(original_init[:200] + "..." if len(original_init) > 200 else original_init)
                updated_init_state = update_initial_state(original_init, action_plan)
            else:
                print("Warning: Could not extract original init state")
                updated_init_state = None
        
        # Debug: Print final robot location after this subtask
        if updated_init_state:
            final_robot_location = extract_robot_location(updated_init_state)
            print(f"Robot final location after subtask {subtask_num}: {final_robot_location}")
        
        return action_plan, updated_init_state, True, robot_init_location, plan_source
    else:
        print("Failed to generate action plan:")
        for line in action_plan:
            print(line)
        
        return action_plan, current_init_state, False, robot_init_location, plan_source

def main():
    # Create timestamped output directory
    output_dir, subtask_problem_dir = create_output_directory()
    
    # Load existing knowledge base
    knowledge_base = load_knowledge_base()
    
    # Define file paths
    base_path = "/home/shaid/Documents/PDDL/problems/real_kitchen_v6/"
    domain_file = os.path.join(base_path, "domain.pddl")
    empty_problem_file = os.path.join(base_path, "empty_problem.pddl")
    
    # Path to manually created subtasks file
    subtasks_file_path = "/home/shaid/Documents/PDDL/subtasks_with_pddl_goals_v2.txt"
    
    # Step 1: Read the manually created subtasks file
    print("\n=== STEP 1: Reading subtasks file ===")
    start_time = time.time()
    
    try:
        plan = read_subtasks_file(subtasks_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create the subtasks_with_pddl_goals.txt file in /home/shaid/Documents/PDDL/")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Step 2: Copy the subtasks file to output directory
    print("\n=== STEP 2: Copying subtasks file to output directory ===")
    subtasks_output_file = os.path.join(output_dir, "subtasks_with_pddl_goals.txt")
    with open(subtasks_output_file, 'w') as f:
        f.write(plan)
    print(f"Subtasks file copied to: {subtasks_output_file}")
    print(f"Read time: {elapsed_time:.2f} seconds")
    
    # Step 3: Parse subtasks and goals
    print("\n=== STEP 3: Parsing subtasks and goals ===")
    subtasks, pddl_goals = parse_subtasks_and_goals(plan)
    
    if not pddl_goals:
        print("No PDDL goals found in the subtasks file.")
        return
    
    print(f"Found {len(subtasks)} subtasks and {len(pddl_goals)} PDDL goals")
    
    for i, (subtask, goal) in enumerate(zip(subtasks, pddl_goals)):
        print(f"Subtask {i+1}: {subtask}")
        print(f"PDDL Goal {i+1}: {goal}")
        print()
    
    # Step 4: Solve all subtasks using knowledge base retrieval and classical planning
    current_init_state = None
    all_action_plans = []
    all_robot_init_locations = []
    plan_sources = []
    kb_retrievals = 0
    classical_calls = 0
    
    for i in range(len(subtasks)):
        subtask_num = i + 1
        subtask = subtasks[i]
        pddl_goal = pddl_goals[i]
        
        action_plan, updated_init_state, success, robot_init_location, plan_source = solve_subtask_with_kb(
            subtask_num, subtask, pddl_goal, domain_file, empty_problem_file, 
            subtask_problem_dir, knowledge_base, current_init_state
        )
        
        if success:
            all_action_plans.append(action_plan)
            all_robot_init_locations.append(robot_init_location)
            plan_sources.append(plan_source)
            current_init_state = updated_init_state
            
            if plan_source == "Knowledge Base":
                kb_retrievals += 1
            else:
                classical_calls += 1
        else:
            print(f"Failed to solve subtask {subtask_num}. Stopping execution.")
            break
    
    # Step 5: Summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total subtasks: {len(subtasks)}")
    print(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
    print(f"Knowledge base retrievals: {kb_retrievals}")
    print(f"Classical planner calls: {classical_calls}")
    print(f"Knowledge base entries after execution: {len(knowledge_base)}")
    
    total_actions = 0
    summary_content = []
    summary_content.append(f"Total subtasks: {len(subtasks)}")
    summary_content.append(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
    summary_content.append(f"Knowledge base retrievals: {kb_retrievals}")
    summary_content.append(f"Classical planner calls: {classical_calls}")
    summary_content.append(f"Knowledge base entries after execution: {len(knowledge_base)}")
    summary_content.append("")
    
    for i, (action_plan, robot_init_location, plan_source) in enumerate(zip(all_action_plans, all_robot_init_locations, plan_sources)):
        subtask_display = f"Subtask-{i+1}: {subtasks[i]}"
        robot_init_display = f"robot_init: {robot_init_location if robot_init_location else 'Unknown'}"
        goal_display = f"Goal-{i+1} (pddl): {pddl_goals[i]}"
        source_display = f"Plan source: {plan_source}"
        actions_display = f"Actions ({len(action_plan)}):"
        
        print(f"\n{subtask_display}")
        print(f"{robot_init_display}")
        print(f"{goal_display}")
        print(f"{source_display}")
        print(f"{actions_display}")
        
        summary_content.append(subtask_display)
        summary_content.append(robot_init_display)
        summary_content.append(goal_display)
        summary_content.append(source_display)
        summary_content.append(actions_display)
        
        for j, action in enumerate(action_plan, 1):
            action_line = f"  {j}. {action}"
            print(action_line)
            summary_content.append(action_line)
        
        summary_content.append("")
        total_actions += len(action_plan)
    
    print(f"\nTotal actions executed: {total_actions}")
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "execution_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_content))
    print(f"Execution summary saved to: {summary_file}")
    
    # Save updated knowledge base
    kb_file = save_knowledge_base(knowledge_base, output_dir)
    
    # Save run configuration
    config_file = os.path.join(output_dir, "run_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Run Configuration\n")
        f.write(f"================\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Subtasks file: {subtasks_file_path}\n")
        f.write(f"Read time: {elapsed_time:.2f} seconds\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Subtask problem directory: {subtask_problem_dir}\n")
        f.write(f"Domain file: {domain_file}\n")
        f.write(f"Problem file: {empty_problem_file}\n")
        f.write(f"Total actions executed: {total_actions}\n")
        f.write(f"Knowledge base retrievals: {kb_retrievals}\n")
        f.write(f"Classical planner calls: {classical_calls}\n")
        f.write(f"Final knowledge base entries: {len(knowledge_base)}\n")
        f.write(f"Updated knowledge base file: {kb_file}\n")
    print(f"Run configuration saved to: {config_file}")
    
    print(f"\nAll output files saved in: {output_dir}")
    print(f"Problem files saved in: {subtask_problem_dir}")
    print(f"Updated knowledge base saved in: {kb_file}")

if __name__ == "__main__":
    main()