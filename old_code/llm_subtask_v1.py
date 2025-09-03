import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def generate_subtasks_with_pddl_goals(task_instructions, predicates_file, actions_file, objects_file, init_state_file, model="gpt-4o", max_tokens=2000):
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

    print("Prompt for OpenAI API:")
    print(prompt)
    
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

def main():
    # Define file paths to the text files
    base_path = "/home/shaid/Documents/kitchen-worlds/plan_reuse/problems/pybullet_kitchen"
    domain_file = os.path.join(base_path, "domain.pddl")
    problem_file = os.path.join(base_path, "empty_problem.pddl")
    predicates_file = os.path.join(base_path, "predicates.txt")
    actions_file = os.path.join(base_path, "actions.txt")
    objects_file = os.path.join(base_path, "objects.txt")
    init_state_file = os.path.join(base_path, "init_state.txt")
    
    # Task instruction
    task_instructions = "Prepare a breakfast by cooking chicken and potato."
    
    # Model and output parameters
    model = "gpt-4o-mini"
    max_tokens = 2000
    output_file = "subtasks_with_pddl_goals.txt"
    print("Model:", model)

    # Print information about the files being used
    print("Using the following files:")
    print(f"- Predicates: {predicates_file}")
    print(f"- Actions: {actions_file}")
    print(f"- Objects: {objects_file}")
    print(f"- Initial State: {init_state_file}")
    
    # Generate the subtasks with PDDL goals
    print("\nGenerating subtasks with PDDL goals...")
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
    
    # Output the plan
    if output_file:
        with open(output_file, 'w') as f:
            f.write(plan)
        print(f"Subtasks with PDDL goals saved to {output_file}")
    
    print("\n--- Generated Subtasks with PDDL Goals ---\n")
    print(plan)
    print(f"\nElapsed Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()