"""PDDL planner integration module."""

import subprocess
from pathlib import Path
from typing import List, Tuple

from config_files.config import PlanReuseConfig
from utils.parsing import parse_planner_output


class PDDLPlanner:
    """Wrapper for Fast Downward classical planner."""
    
    def __init__(self, config: PlanReuseConfig):
        self.config = config
        self.fd_script = config.fast_downward_dir / "fast-downward.py"
        
        if not self.fd_script.exists():
            raise FileNotFoundError(f"Fast Downward script not found: {self.fd_script}")
    
    def solve(self, domain_file: Path, problem_file: Path) -> Tuple[List[str], bool]:
        """Run the Fast Downward classical planner and extract the action plan."""
        command = [
            "python3", str(self.fd_script),
            str(domain_file),
            str(problem_file),
            "--search", self.config.search_option
        ]
        
        print(f"Running classical planner with command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=self.config.planner_timeout
            )
            
            plan_lines, success = parse_planner_output(result.stdout)
            return plan_lines, success
                
        except subprocess.TimeoutExpired:
            return [f"Planner timed out after {self.config.planner_timeout} seconds."], False
        except Exception as e:
            return [f"Error running planner: {str(e)}"], False