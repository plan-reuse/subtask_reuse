"""Knowledge base module for storing and retrieving abstract plans."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config_files.config import PlanReuseConfig


class KnowledgeBase:
    """Manages storage and retrieval of abstract plans."""
    
    def __init__(self, config: PlanReuseConfig):
        self.config = config
        self.kb_data: Dict[Tuple[str, str], List[str]] = {}
        
        if config.kb_enabled:
            self.load()
    
    def load(self) -> None:
        """Load the existing knowledge base from file."""
        print(f"Attempting to load knowledge base from: {self.config.kb_file}")
        print(f"File exists: {self.config.kb_file.exists()}")
        
        if not self.config.kb_file.exists():
            print("No existing knowledge base found. Starting with empty knowledge base.")
            return
        
        # Read and execute the knowledge base file
        with open(self.config.kb_file, 'r') as f:
            content = f.read()
        
        print(f"Knowledge base file content length: {len(content)} characters")
        
        # Extract the knowledge_base dictionary from the file
        local_vars = {}
        exec(content, {}, local_vars)
        self.kb_data = local_vars.get('knowledge_base', {})
        
        print(f"Loaded knowledge base with {len(self.kb_data)} entries")
        if self.kb_data:
            print("Knowledge base keys:")
            for key in self.kb_data.keys():
                print(f"  - {key}")
    
    def save(self, output_dir: Path) -> Path:
        """Save knowledge base to a file in dictionary format."""
        kb_file = output_dir / "knowledge_base.txt"
        print(f"Saving knowledge base to: {kb_file}")
        
        with open(kb_file, 'w') as f:
            f.write("knowledge_base = {\n")
            
            for i, (key, actions) in enumerate(self.kb_data.items()):
                goal_str, robot_init_str = key
                
                f.write("    (\n")
                f.write(f'        "{goal_str}",\n')
                f.write(f'        "{robot_init_str}"\n')
                f.write("    ): [\n")
                
                for action in actions:
                    f.write(f'        "{action}",\n')
                
                f.write("    ]")
                
                # Add comma if not the last entry
                if i < len(self.kb_data) - 1:
                    f.write(",")
                f.write("\n")
            
            f.write("}\n")
        
        print(f"Knowledge base saved to: {kb_file}")
        return kb_file
    
    def retrieve(self, search_key: Tuple[str, str]) -> Optional[List[str]]:
        """Retrieve abstract plan for the given search key."""
        return self.kb_data.get(search_key)
    
    def store(self, search_key: Tuple[str, str], abstract_plan: List[str]) -> None:
        """Store abstract plan with the given search key."""
        self.kb_data[search_key] = abstract_plan
        print(f"Added new knowledge base entry:")
        print(f"Key: {search_key}")
        print(f"Abstract plan: {abstract_plan}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        return {
            "total_entries": len(self.kb_data),
            "total_actions": sum(len(actions) for actions in self.kb_data.values())
        }
    
    def list_keys(self) -> List[Tuple[str, str]]:
        """List all keys in the knowledge base."""
        return list(self.kb_data.keys())
    
    def contains(self, search_key: Tuple[str, str]) -> bool:
        """Check if search key exists in knowledge base."""
        return search_key in self.kb_data