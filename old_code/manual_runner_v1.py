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

def validate_pddl_goal(pddl_goal):
    """Validate PDDL goal for logical contradictions"""
    # Extract all predicates from the goal
    predicates = re.findall(r'\([^)]+\)', pddl_goal)
    
    holding_objects = set()
    location_objects = {}
    
    for pred in predicates:
        pred = pred.strip('()')
        parts = pred.split()
        
        if len(parts) >= 3:
            predicate_name = parts[0]
            
            # Track holding relationships
            if predicate_name == "holding":
                robot, obj = parts[1], parts[2]
                holding_objects.add(obj)
            
            # Track location relationships
            elif predicate_name in ["inside", "at", "on_stovetop", "in_cookware", "on_utensil"]:
                obj, location = parts[1], parts[2]
                location_objects[obj] = (predicate_name, location)
    
    # Check for contradictions
    contradictions = []
    for obj in holding_objects:
        if obj in location_objects:
            pred_name, location = location_objects[obj]
            contradictions.append(f"Object '{obj}' cannot be both held and {pred_name} {location}")
    
    return len(contradictions) == 0, contradictions

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
            effects.append(f"(not (at_robot {robot} {from_loc}))")
            effects.append(f"(at_robot {robot} {to_loc})")
    
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
            effects.append(f"(holding {robot} {obj})")
            effects.append(f"(not (at {obj} {location}))")
            effects.append(f"(not (handempty {robot}))")
    
    # Putdown to region
    elif action.startswith("putdown_to_region "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, location = parts[1], parts[2], parts[3]
            effects.append(f"(at {obj} {location})")
            effects.append(f"(handempty {robot})")
            effects.append(f"(not (holding {robot} {obj}))")
    
    # Pickup from container
    elif action.startswith("pickup_from_container "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, container = parts[1], parts[2], parts[3]
            effects.append(f"(holding {robot} {obj})")
            effects.append(f"(not (inside {obj} {container}))")
            effects.append(f"(not (handempty {robot}))")
    
    # Putdown to container
    elif action.startswith("putdown_to_container "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, container = parts[1], parts[2], parts[3]
            effects.append(f"(inside {obj} {container})")
            effects.append(f"(handempty {robot})")
            effects.append(f"(not (holding {robot} {obj}))")
    
    # Put food in cookware
    elif action.startswith("put_food_in_cookware "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
            effects.append(f"(in_cookware {food} {cookware})")
            effects.append(f"(handempty {robot})")
            effects.append(f"(not (holding {robot} {food}))")
    
    # Take food from cookware
    elif action.startswith("take_food_from_cookware "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
            effects.append(f"(holding {robot} {food})")
            effects.append(f"(not (in_cookware {food} {cookware}))")
            effects.append(f"(not (handempty {robot}))")
    
    # Place food on utensil
    elif action.startswith("place_food_on_utensil "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, utensil, location = parts[1], parts[2], parts[3], parts[4]
            effects.append(f"(on_utensil {food} {utensil})")
            effects.append(f"(handempty {robot})")
            effects.append(f"(not (holding {robot} {food}))")
    
    # Take food from utensil
    elif action.startswith("take_food_from_utensil "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, utensil, location = parts[1], parts[2], parts[3], parts[4]
            effects.append(f"(holding {robot} {food})")
            effects.append(f"(not (on_utensil {food} {utensil}))")
            effects.append(f"(not (handempty {robot}))")
    
    # Place cookware on stovetop
    elif action.startswith("place_cookware_on_stovetop "):
        parts = action.split()
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            effects.append(f"(on_stovetop {cookware} {stovetop})")
            effects.append(f"(handempty {robot})")
            effects.append(f"(not (holding {robot} {cookware}))")
    
    # Remove cookware from stovetop
    elif action.startswith("remove_cookware_from_stovetop "):
        parts = action.split()
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            effects.append(f"(holding {robot} {cookware})")
            effects.append(f"(not (on_stovetop {cookware} {stovetop}))")
            effects.append(f"(not (handempty {robot}))")
    
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

def extract_robot_location(init_state_text, robot_name="stretch_robot"):
    """Extract the robot's location from the initial state"""
    predicates = extract_predicates_from_init(init_state_text)
    
    for predicate in predicates:
        if predicate.startswith(f"(at_robot {robot_name}"):
            return predicate
    
    return None

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
                if predicate_to_remove in init_predicates:
                    init_predicates.remove(predicate_to_remove)
                    print(f"  Removed: {predicate_to_remove}")
                else:
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
    
    # Debug: Print the complete updated problem file
    print(f"\n--- Contents of {output_file} ---")
    with open(output_file, 'r') as f:
        content = f.read()
    print(content)
    print("--- End of file ---")

def solve_subtask(subtask_num, subtask, pddl_goal, domain_file, problem_file, subtask_problem_dir, current_init_state=None):
    """Solve a single subtask and return the action plan, updated initial state, and robot's initial location"""
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
    
    # Validate PDDL goal before planning
    is_valid, errors = validate_pddl_goal(pddl_goal)
    if not is_valid:
        print(f"❌ Invalid PDDL goal detected:")
        for error in errors:
            print(f"   - {error}")
        return [], current_init_state, False, robot_init_location
    
    # Create problem file for this subtask in the subtask_problem directory
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
    
    print(f"\n--- Action Plan for Subtask {subtask_num} ---")
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
        
        return action_plan, updated_init_state, True, robot_init_location
    else:
        print("Failed to generate action plan:")
        for line in action_plan:
            print(line)
        
        return action_plan, current_init_state, False, robot_init_location

def main():
    # Create timestamped output directory
    output_dir, subtask_problem_dir = create_output_directory()
    
    # Define file paths
    base_path = "/home/shaid/Documents/PDDL/problems/real_kitchen_v6/"
    domain_file = os.path.join(base_path, "domain.pddl")
    empty_problem_file = os.path.join(base_path, "empty_problem.pddl")
    
    # Path to manually created subtasks file
    subtasks_file_path = "/home/shaid/Documents/PDDL/subtasks_with_pddl_goals.txt"
    
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
    
    # Validate all goals before starting
    print("\n=== STEP 4: Validating PDDL goals ===")
    for i, goal in enumerate(pddl_goals):
        is_valid, errors = validate_pddl_goal(goal)
        if not is_valid:
            print(f"⚠️  Goal {i+1} has contradictions:")
            for error in errors:
                print(f"   - {error}")
            print("Please fix the goal before proceeding.")
        else:
            print(f"✅ Goal {i+1} is valid")
    
    for i, (subtask, goal) in enumerate(zip(subtasks, pddl_goals)):
        print(f"Subtask {i+1}: {subtask}")
        print(f"PDDL Goal {i+1}: {goal}")
        print()
    
    # Step 5: Solve all subtasks
    current_init_state = None
    all_action_plans = []
    all_robot_init_locations = []
    
    for i in range(len(subtasks)):
        subtask_num = i + 1
        subtask = subtasks[i]
        pddl_goal = pddl_goals[i]
        
        action_plan, updated_init_state, success, robot_init_location = solve_subtask(
            subtask_num, subtask, pddl_goal, domain_file, empty_problem_file, subtask_problem_dir, current_init_state
        )
        
        if success:
            all_action_plans.append(action_plan)
            all_robot_init_locations.append(robot_init_location)
            current_init_state = updated_init_state
        else:
            print(f"Failed to solve subtask {subtask_num}. Stopping execution.")
            break
    
    # Step 6: Summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total subtasks: {len(subtasks)}")
    print(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
    
    total_actions = 0
    summary_content = []
    summary_content.append(f"Total subtasks: {len(subtasks)}")
    summary_content.append(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
    summary_content.append("")
    
    for i, (action_plan, robot_init_location) in enumerate(zip(all_action_plans, all_robot_init_locations)):
        subtask_display = f"Subtask-{i+1}: {subtasks[i]}"
        robot_init_display = f"robot_init: {robot_init_location if robot_init_location else 'Unknown'}"
        goal_display = f"Goal-{i+1} (pddl): {pddl_goals[i]}"
        actions_display = f"Actions ({len(action_plan)}):"
        
        print(f"\n{subtask_display}")
        print(f"{robot_init_display}")
        print(f"{goal_display}")
        print(f"{actions_display}")
        
        summary_content.append(subtask_display)
        summary_content.append(robot_init_display)
        summary_content.append(goal_display)
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
    print(f"Run configuration saved to: {config_file}")
    
    print(f"\nAll output files saved in: {output_dir}")
    print(f"Problem files saved in: {subtask_problem_dir}")

if __name__ == "__main__":
    main()