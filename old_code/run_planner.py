import subprocess
import os
import re

# === CONFIGURATION ===
PDDL_ROOT = "/home/shaid/Documents/kitchen-worlds/plan_reuse"
FAST_DOWNWARD_DIR = os.path.join(PDDL_ROOT, "downward")
PROBLEMS_DIR = os.path.join(PDDL_ROOT, "problems")

# === SELECT PROBLEM NAME HERE ===
PROBLEM_NAME = "pybullet_kitchen"


# === SEARCH OPTION ===
SEARCH_OPTION = 'astar(blind())'

def run_fast_downward(problem_name):
    domain_file = os.path.join(PROBLEMS_DIR, problem_name, "domain.pddl")
    problem_file = os.path.join(PROBLEMS_DIR, problem_name, "problem.pddl")
    fd_script = os.path.join(FAST_DOWNWARD_DIR, "fast-downward.py")

    command = [
        "python3", fd_script,
        domain_file,
        problem_file,
        "--search", SEARCH_OPTION
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    

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
    

    print("=== PLAN ===")
    for line in plan_lines:
        print(line)
    

    if not plan_lines:
        if "Solution found" in result.stdout:
            print("A plan was found but couldn't be extracted properly.")
        else:
            print("No plan found.")

if __name__ == "__main__":
    run_fast_downward(PROBLEM_NAME)