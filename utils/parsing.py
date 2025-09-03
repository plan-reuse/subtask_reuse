"""Parsing utilities for PDDL and text processing."""

import re
from typing import List, Tuple, Set


def parse_subtasks_and_goals(plan_text: str) -> Tuple[List[str], List[str]]:
    """Parse the generated plan text to extract subtasks and PDDL goals."""
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


def extract_predicates_from_init(init_state_text: str) -> Set[str]:
    """Extract all predicates from an init state text."""
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


def extract_robot_location(init_state_text: str, robot_name: str = "pr2") -> str:
    """Extract the robot's location from the initial state."""
    predicates = extract_predicates_from_init(init_state_text)
    
    for predicate in predicates:
        # Look for any robot location predicate, not just with the specific robot name
        if predicate.startswith("(at_robot"):
            parts = predicate.strip('()').split()
            if len(parts) >= 3:
                # Extract the actual robot name and location
                actual_robot = parts[1]
                location = parts[2]
                # Return in standardized format using "pr2"
                return f"(at_robot {robot_name} {location})"
    
    return None


def create_problem_file_with_goal(empty_problem_file: str, output_problem_file: str, pddl_goal: str) -> None:
    """Create a new problem file with the specified PDDL goal."""
    with open(empty_problem_file, 'r') as f:
        content = f.read()
    
    # Replace the empty goal section with the actual goal
    goal_section = f"(:goal\n    {pddl_goal}\n  )"
    updated_content = re.sub(r'\(:goal\s*\n\s*\)', goal_section, content)
    
    with open(output_problem_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Created problem file with goal: {output_problem_file}")


def create_updated_problem_file(base_problem_file: str, updated_init_state: str, 
                              pddl_goal: str, output_file: str) -> None:
    """Create a new problem file with updated initial state and goal."""
    with open(base_problem_file, 'r') as f:
        content = f.read()
    
    # Find the current init section and replace it
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


def extract_original_init_state(problem_file: str) -> str:
    """Extract the original init state from a problem file."""
    with open(problem_file, 'r') as f:
        content = f.read()
    
    # Extract original init state with better parsing
    init_match = re.search(r'\(:init\s*\n(.*?)\n\s*\)', content, re.DOTALL)
    if init_match:
        # Reconstruct the full init state
        init_content = init_match.group(1).strip()
        return f"(:init\n{init_content}\n  )"
    else:
        raise ValueError("Could not extract original init state from problem file")


def parse_planner_output(stdout: str) -> Tuple[List[str], bool]:
    """Parse Fast Downward planner output to extract action plan."""
    plan_lines = []
    stdout_lines = stdout.split('\n')
    
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
        if "Solution found" in stdout:
            return ["A plan was found but couldn't be extracted properly."], False
        else:
            return ["No plan found."], False