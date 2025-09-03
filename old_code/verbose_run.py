import subprocess
import os

# === CONFIGURATION ===
PDDL_ROOT = "/home/shaid/Documents/PDDL"
FAST_DOWNWARD_DIR = os.path.join(PDDL_ROOT, "downward")
PROBLEMS_DIR = os.path.join(PDDL_ROOT, "problems")

# === SELECT PROBLEM NAME HERE ===
# PROBLEM_NAME = "hello_world"
PROBLEM_NAME = "test"
# PROBLEM_NAME = "blocks_world"

# === SEARCH OPTION ===
SEARCH_OPTION = 'lazy_greedy([ff()], preferred=[ff()])'
# SEARCH_OPTION = 'astar(hmax())'
# SEARCH_OPTION = 'astar(lmcut())'
# SEARCH_OPTION = 'astar(blind())'

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

    print(f"Running command: {' '.join(command)}\n")
    result = subprocess.run(command, capture_output=True, text=True)

    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)

if __name__ == "__main__":
    run_fast_downward(PROBLEM_NAME)