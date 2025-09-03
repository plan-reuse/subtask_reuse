import os
import time
import subprocess
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def generate_subtasks_with_pddl_goals(task_instructions, predicates_file, actions_file, objects_file, init_state_file, model="gpt-4o", max_tokens=2000):
    """Generate natural language subtasks and corresponding PDDL goals using OpenAI API"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    
    # Read the text files directly
    with open(predicates_file, 'r') as f:
        predicates_content = f.read()
    
    with open(actions_file, 'r') as f:
        actions_content = f.read()
    
    with open(objects_file, 'r') as f:
        objects_content = f.read()
    
    with open(init_state_file, 'r') as f:
        init_state_content = f.read()
    
    # Create the prompt for the API
    prompt = f"""
    Task: {task_instructions}
    
    I will provide you with information about a kitchen domain, including available predicates, actions, objects, and their initial states. Your job is to break down the main task into natural language subtasks that need to be completed in sequence, and provide the corresponding PDDL goal for each subtask.
    
    Available Predicates:
    {predicates_content}
    
    Available Actions:
    {actions_content}
    
    Objects by Type:
    {objects_content}
    
    Initial State:
    {init_state_content}
    
    Based on the task description, available actions, object list, and initial states, generate a list of subtasks in natural language that would need to be completed to achieve the goal.
    
    Requirements:
    1. Each subtask should be a clear, concise instruction in natural language in a single sentence
    2. Explicitly include the objects involved in each subtask
    3. Make sure the subtasks follow a logical sequence to achieve the goal
    4. Only include actions that are possible given the predicates and actions available
    5. For each natural language subtask, provide the corresponding PDDL goal that represents the desired state after completing that subtask
    
    Format the response as follows:
    Subtask 1: [Natural language subtask]
    PDDL Goal 1: [PDDL goal expression]
    
    Subtask 2: [Natural language subtask]
    PDDL Goal 2: [PDDL goal expression]
    
    ...and so on.
    """

    # print("Prompt for OpenAI API:")
    # print(prompt)
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI specialized in planning and PDDL. Your task is to generate natural language subtask plans based on high-level instructions and available actions, along with their corresponding PDDL goals."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    
    # Extract and return the generated plan
    plan = response.choices[0].message.content
    return plan

def parse_subtasks_and_goals(plan_text):
    """Parse the generated plan text to extract subtasks and PDDL goals"""
    lines = plan_text.strip().split('\n')
    subtasks = []
    pddl_goals = []
    
    current_subtask = ""
    current_goal = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith("Subtask"):
            if current_subtask and current_goal:
                subtasks.append(current_subtask)
                pddl_goals.append(current_goal)
            current_subtask = line.split(":", 1)[1].strip() if ":" in line else ""
            current_goal = ""
        elif line.startswith("PDDL Goal"):
            current_goal = line.split(":", 1)[1].strip() if ":" in line else ""
    
    # Add the last pair
    if current_subtask and current_goal:
        subtasks.append(current_subtask)
        pddl_goals.append(current_goal)
    
    return subtasks, pddl_goals

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

def main():
    # Define file paths
    base_path = "/home/shaid/Documents/PDDL/problems/real_kitchen_v6/"
    domain_file = os.path.join(base_path, "domain.pddl")
    empty_problem_file = os.path.join(base_path, "empty_problem.pddl")
    predicates_file = os.path.join(base_path, "predicates.txt")
    actions_file = os.path.join(base_path, "actions.txt")
    objects_file = os.path.join(base_path, "objects.txt")
    init_state_file = os.path.join(base_path, "init_state.txt")
    
    # Task instruction
    task_instructions = "Prepare a breakfast by cooking chicken and potato."
    
    # Model and output parameters
    model = "gpt-4o-mini"
    max_tokens = 2000
    subtasks_output_file = "subtasks_with_pddl_goals.txt"
    problem_with_goal_file = os.path.join(base_path, "problem_with_first_goal.pddl")
    
    
    # Step 1: Generate the subtasks with PDDL goals
    print("\n=== STEP 1: Generating subtasks with PDDL goals ===")
    start_time = time.time()
    
    plan = generate_subtasks_with_pddl_goals(
        task_instructions,
        predicates_file,
        actions_file,
        objects_file,
        init_state_file,
        model=model,
        max_tokens=max_tokens
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Step 2: Save and display the generated subtasks
    print("\n=== STEP 2: Saving and displaying subtasks ===")
    with open(subtasks_output_file, 'w') as f:
        f.write(plan)
    print(f"Subtasks with PDDL goals saved to {subtasks_output_file}")
    
    # print("\n--- Generated Subtasks with PDDL Goals ---")
    # print(plan)
    print(f"Generation time: {elapsed_time:.2f} seconds")
    
    # Step 3: Parse subtasks and goals
    print("\n=== STEP 3: Parsing subtasks and goals ===")
    subtasks, pddl_goals = parse_subtasks_and_goals(plan)
    
    if not pddl_goals:
        print("No PDDL goals found in the generated plan.")
        return
    
    print(f"Found {len(subtasks)} subtasks and {len(pddl_goals)} PDDL goals")
    for i, (subtask, goal) in enumerate(zip(subtasks, pddl_goals)):
        print(f"Subtask {i+1}: {subtask}")
        print(f"PDDL Goal {i+1}: {goal}")
        print()
    
    # Step 4: Create problem file with first PDDL goal
    print("\n=== STEP 4: Creating problem file with first PDDL goal ===")
    first_pddl_goal = pddl_goals[0]
    print(f"First PDDL goal: {first_pddl_goal}")
    
    create_problem_file_with_goal(empty_problem_file, problem_with_goal_file, first_pddl_goal)
    
    # Step 5: Run classical planner for the first goal
    print("\n=== STEP 5: Running classical planner for first goal ===")
    action_plan, success = run_classical_planner(domain_file, problem_with_goal_file)
    
    print("\n--- Action Plan for First Goal ---")
    if success:
        print(f"Successfully generated action plan with {len(action_plan)} steps:")
        for i, action in enumerate(action_plan, 1):
            print(f"{i}. {action}")
    else:
        print("Failed to generate action plan:")
        for line in action_plan:
            print(line)
    
    print("\n=== SUMMARY ===")
    print(f"Task: {task_instructions}")
    print(f"First subtask: {subtasks[0] if subtasks else 'N/A'}")
    print(f"First PDDL goal: {first_pddl_goal}")
    print(f"Action plan generated: {'Yes' if success else 'No'}")
    if success:
        print(f"Number of actions: {len(action_plan)}")

if __name__ == "__main__":
    main()