"""Task generation module supporting both LLM and manual input."""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

from config_files.config import PlanReuseConfig
from utils.parsing import parse_subtasks_and_goals
from utils.file_ops import read_file


class TaskGenerator:
    """Generates subtasks using LLM or manual input."""
    
    def __init__(self, config: PlanReuseConfig):
        self.config = config
        self.client = None
        
        if config.use_llm:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=api_key)
    
    def generate_subtasks(self, task_instruction: str) -> Tuple[List[str], List[str]]:
        """Generate subtasks and PDDL goals for the given task instruction."""
        if self.config.use_llm:
            return self._generate_with_llm(task_instruction)
        elif self.config.fallback_file:
            return self._load_from_file(self.config.fallback_file)
        else:
            raise ValueError("No generation method available. Either enable LLM or provide fallback file.")
    
    def _generate_with_llm(self, task_instruction: str) -> Tuple[List[str], List[str]]:
        """Generate subtasks using OpenAI API."""
        print("Generating subtasks using LLM...")
        
        # Read required files
        predicates_content = read_file(self.config.predicates_file)
        actions_content = read_file(self.config.actions_file)
        objects_content = read_file(self.config.objects_file)
        init_state_content = read_file(self.config.init_state_file)
        
        # Read prompt template
        prompt_template = read_file(self.config.llm_prompt_template_file)
        
        # Create the prompt by substituting variables
        prompt = prompt_template.format(
            task_instruction=task_instruction,
            predicates_content=predicates_content,
            actions_content=actions_content,
            objects_content=objects_content,
            init_state_content=init_state_content
        )
        
        # Read system prompt
        system_prompt = read_file(self.config.llm_system_prompt_file)
        
        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.max_tokens
        )
        
        # Extract and parse the generated plan
        plan_text = response.choices[0].message.content
        subtasks, pddl_goals = parse_subtasks_and_goals(plan_text)
        
        print(f"Generated {len(subtasks)} subtasks using LLM")
        return subtasks, pddl_goals
    
    def _load_from_file(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """Load subtasks from a manual file."""
        print(f"Loading subtasks from file: {file_path}")
        
        content = read_file(file_path)
        subtasks, pddl_goals = parse_subtasks_and_goals(content)
        
        print(f"Loaded {len(subtasks)} subtasks from file")
        return subtasks, pddl_goals
    
    def save_generated_subtasks(self, subtasks: List[str], pddl_goals: List[str], output_file: Path) -> None:
        """Save generated subtasks to a file."""
        content_lines = []
        
        for i, (subtask, goal) in enumerate(zip(subtasks, pddl_goals), 1):
            content_lines.append(f"Subtask {i}: {subtask}")
            content_lines.append(f"PDDL Goal {i}: {goal}")
            content_lines.append("")  # Empty line for readability
        
        content = "\n".join(content_lines)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"Subtasks saved to: {output_file}")
    
    def main(self):
        """Main function to test TaskGenerator standalone."""
        print("TaskGenerator standalone test")
        print(f"Using LLM: {self.config.use_llm}")
        print(f"Model: {self.config.model if self.config.use_llm else 'N/A'}")
        print(f"Fallback file: {self.config.fallback_file}")
        
        # Test task instruction
        test_task = "Prepare a breakfast by cooking chicken"
        print(f"\nTest task: {test_task}")
        
        try:
            subtasks, pddl_goals = self.generate_subtasks(test_task)
            
            print(f"\nGenerated {len(subtasks)} subtasks:")
            for i, subtask in enumerate(subtasks, 1):
                print(f"  {i}. {subtask}")
            
            print(f"\nGenerated {len(pddl_goals)} PDDL goals:")
            for i, goal in enumerate(pddl_goals, 1):
                print(f"  {i}. {goal}")

            output_file = self.config.output_dir / "generated_subtasks.txt"
            self.save_generated_subtasks(subtasks, pddl_goals, output_file)
                
        except Exception as e:
            print(f"Error generating subtasks: {e}")


if __name__ == "__main__":
    # Example usage for standalone testing
    from config_files.config import PlanReuseConfig
    
    # Create a default config for testing
    config = PlanReuseConfig()
    # config = PlanReuseConfig(use_llm=False, fallback_file=Path("sample_fallback.txt"))
    
    # Initialize TaskGenerator
    task_gen = TaskGenerator(config)
    
    # Run main function
    task_gen.main()