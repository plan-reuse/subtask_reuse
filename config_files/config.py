"""Configuration module for plan reuse system."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlanReuseConfig:
    """Configuration for the plan reuse system."""
    
    # Generation mode
    use_llm: bool = True
    fallback_file: Optional[Path] = None
    
    # LLM settings
    model: str = "gpt-4o"
    max_tokens: int = 2000
    
    # Knowledge base
    kb_enabled: bool = True
    kb_file: Path = Path("output/knowledge_base.txt")
    
    # Planning settings
    planner_timeout: int = 60
    search_option: str = "astar(blind())"
    
    # Paths
    domain_file: Path = Path("problems/pybullet_kitchen/domain.pddl")
    problem_file: Path = Path("problems/pybullet_kitchen/empty_problem.pddl")
    predicates_file: Path = Path("problems/pybullet_kitchen/predicates.txt")
    actions_file: Path = Path("problems/pybullet_kitchen/actions.txt")
    objects_file: Path = Path("problems/pybullet_kitchen/objects.txt")
    init_state_file: Path = Path("problems/pybullet_kitchen/init_state.txt")
    output_dir: Path = Path("output")
    fast_downward_dir: Path = Path("downward")
    
    # LLM prompt files
    # llm_prompt_template_file: Path = Path("config_files/llm_prompt_template.txt")
    llm_prompt_template_file: Path = Path("config_files/llm_prompt_template.txt")
    llm_system_prompt_file: Path = Path("config_files/llm_system_prompt.txt")
    
    # Task settings
    task_instruction: str = "cook chicken with salt"
    
    def __post_init__(self):
        """Convert string paths to Path objects and resolve relative paths."""
        # Convert all path attributes to Path objects and make them absolute
        base_path = Path("/home/shaid/Documents/kitchen-worlds/plan_reuse")
        
        # Special handling for kb_file - always place in base output directory (not timestamped runs)
        kb_filename = Path(self.kb_file).name
        
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, (str, Path)) and field_name.endswith(('_file', '_dir')):
                if field_name == 'kb_file':
                    # Skip kb_file here, handle it separately
                    continue
                path_obj = Path(value)
                if not path_obj.is_absolute():
                    path_obj = base_path / path_obj
                setattr(self, field_name, path_obj)
        
        # Now set kb_file to be in the output directory
        self.kb_file = self.output_dir / kb_filename
        
        # Ensure kb_file points to base output directory, not any timestamped subdirectories
        # This prevents the issue where each run creates a new timestamped directory
        # but the knowledge base should persist across runs
    
    @classmethod
    def from_environment(cls) -> 'PlanReuseConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("USE_LLM"):
            config.use_llm = os.getenv("USE_LLM").lower() == "true"
        
        if os.getenv("FALLBACK_FILE"):
            config.fallback_file = Path(os.getenv("FALLBACK_FILE"))
        
        if os.getenv("OPENAI_MODEL"):
            config.model = os.getenv("OPENAI_MODEL")
        
        if os.getenv("KB_ENABLED"):
            config.kb_enabled = os.getenv("KB_ENABLED").lower() == "true"
        
        if os.getenv("TASK_INSTRUCTION"):
            config.task_instruction = os.getenv("TASK_INSTRUCTION")
        
        return config
    
    def validate(self) -> None:
        """Validate the configuration."""
        required_files = [
            self.domain_file,
            self.problem_file,
        ]
        
        if self.use_llm:
            required_files.extend([
                self.predicates_file,
                self.actions_file,
                self.objects_file,
                self.init_state_file,
                self.llm_prompt_template_file,
                self.llm_system_prompt_file,
            ])
        
        if self.fallback_file:
            required_files.append(self.fallback_file)
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Required files not found: {missing_files}")
        
        if not self.fast_downward_dir.exists():
            raise FileNotFoundError(f"Fast Downward directory not found: {self.fast_downward_dir}")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    """Test configuration file path resolution."""
    print("Testing PlanReuseConfig file paths:")
    print("=" * 50)
    
    config = PlanReuseConfig()
    
    # Print all file and directory paths
    print("File and directory paths:")
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        if isinstance(value, Path) and field_name.endswith(('_file', '_dir')):
            exists = "✓" if value.exists() else "✗"
            print(f"  {field_name:25}: {exists} {value}")
    
    print("\nSpecial checks:")
    print(f"  kb_file exists:         {'✓' if config.kb_file.exists() else '✗'} {config.kb_file}")
    print(f"  kb_file parent exists:  {'✓' if config.kb_file.parent.exists() else '✗'} {config.kb_file.parent}")
    print(f"  Current working dir:    {Path.cwd()}")
    