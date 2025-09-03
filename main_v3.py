"""Unified main entry point for the plan reuse system with SimulatorExecutor v4 integration."""

import time
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add simulation paths
sys.path.extend([
    '../pybullet_planning',
    '../pddlstream', 
    '../lisdf'
])

from config_files.config import PlanReuseConfig
from core.task_generator import TaskGenerator
from core.knowledge_base import KnowledgeBase
from core.planner import PDDLPlanner
from core.state_manager import StateManager
from core.abstraction_v2 import PlanAbstractor
from core.simulator_executor_v6 import SimulatorExecutor
from utils.file_ops import create_output_directory
from utils.parsing import create_problem_file_with_goal, create_updated_problem_file, extract_robot_location


class PlanReuseSystemV3:

    def __init__(self, config: PlanReuseConfig, use_simulation: bool = True):
        self.config = config
        self.use_simulation = use_simulation
        
        # Initialize components
        self.task_generator = TaskGenerator(config)
        self.knowledge_base = KnowledgeBase(config) if config.kb_enabled else None
        self.planner = PDDLPlanner(config)
        self.state_manager = StateManager()
        self.abstractor = PlanAbstractor()
        
        # Simulation executor (handles its own initialization)
        self.executor = None
        
        # Statistics
        self.kb_retrievals = 0
        self.classical_calls = 0
    
    def solve_subtask(self, subtask_num: int, subtask: str, pddl_goal: str, 
                     subtask_problem_dir: Path, current_init_state: str = None) -> Tuple[List[str], str, bool, str, str]:
        """Solve a single subtask using knowledge base retrieval or classical planning."""
        print(f"\n{'='*50}")
        print(f"SOLVING SUBTASK {subtask_num}")
        print(f"{'='*50}")
        print(f"Subtask: {subtask}")
        print(f"PDDL Goal: {pddl_goal}")
        
        # Extract robot's initial location for this subtask
        if current_init_state:
            robot_init_location = extract_robot_location(current_init_state)
        else:
            # For first subtask, load the original init state
            original_init = self.state_manager.load_original_state(str(self.config.problem_file))
            robot_init_location = extract_robot_location(original_init)
        
        print(f"Robot initial location: {robot_init_location}")
        
        # Create search key using abstracted goal and robot initial location
        search_key = self.abstractor.create_search_key(pddl_goal, robot_init_location)
        print(f"Search key: {search_key}")
        
        # Try knowledge base retrieval first if enabled
        action_plan = None
        plan_source = "Classical Planner"
        
        if self.knowledge_base and self.knowledge_base.contains(search_key):
            print("🎯 Found matching entry in knowledge base!")
            abstract_plan = self.knowledge_base.retrieve(search_key)
            print(f"Retrieved abstract plan:")
            for i, action in enumerate(abstract_plan, 1):
                print(f"  {i}. {action}")
            
            # Instantiate the abstract plan
            print("\n--- Instantiating abstract plan ---")
            action_plan = self.abstractor.instantiate_abstract_plan(abstract_plan, pddl_goal, robot_init_location)
            print(f"Instantiated plan:")
            for i, action in enumerate(action_plan, 1):
                print(f"  {i}. {action}")
            
            success = True
            plan_source = "Knowledge Base"
            self.kb_retrievals += 1
        
        # Fall back to classical planner if no KB match
        if action_plan is None:
            print("❌ No matching entry found in knowledge base. Using classical planner.")
            
            # Create problem file for this subtask
            if current_init_state:
                # Use updated initial state
                temp_problem_file = subtask_problem_dir / f"temp_problem_subtask_{subtask_num}.pddl"
                create_updated_problem_file(str(self.config.problem_file), current_init_state, pddl_goal, str(temp_problem_file))
                problem_file_to_use = temp_problem_file
            else:
                # Use original problem file with goal
                problem_file_to_use = subtask_problem_dir / f"temp_problem_subtask_{subtask_num}.pddl"
                create_problem_file_with_goal(str(self.config.problem_file), str(problem_file_to_use), pddl_goal)
            
            # Run planner
            action_plan, success = self.planner.solve(self.config.domain_file, problem_file_to_use)
            self.classical_calls += 1
            
            # Store in knowledge base if successful and KB is enabled
            if success and self.knowledge_base:
                print("\n--- Adding new entry to knowledge base ---")
                abstract_plan = []
                for action in action_plan:
                    abstract_action_str = self.abstractor.abstract_action(action)
                    abstract_plan.append(abstract_action_str)
                
                self.knowledge_base.store(search_key, abstract_plan)
        
        print(f"\n--- Action Plan for Subtask {subtask_num} (from {plan_source}) ---")
        if success:
            print(f"Successfully generated action plan with {len(action_plan)} steps:")
            for i, action in enumerate(action_plan, 1):
                print(f"{i}. {action}")
            
            # Update initial state for next subtask
            if current_init_state:
                updated_init_state = self.state_manager.update_state(action_plan, current_init_state)
            else:
                # For first subtask, use the original state we loaded
                updated_init_state = self.state_manager.update_state(action_plan, self.state_manager.get_current_state())
            
            # Debug: Print final robot location after this subtask
            if updated_init_state:
                final_robot_location = extract_robot_location(updated_init_state)
                print(f"Robot final location after subtask {subtask_num}: {final_robot_location}")
            
            return action_plan, updated_init_state, True, robot_init_location, plan_source
        else:
            print("Failed to generate action plan:")
            for line in action_plan:
                print(line)
            
            return action_plan, current_init_state, False, robot_init_location, plan_source
    
    def run(self) -> None:
        """Run the complete plan reuse system with SimulatorExecutor v4."""
        print("="*60)
        print("PLAN REUSE SYSTEM V3 WITH SIMULATOR EXECUTOR V4")
        print("="*60)
        print(f"Configuration:")
        print(f"  Task: {self.config.task_instruction}")
        print(f"  Use LLM: {self.config.use_llm}")
        print(f"  Fallback file: {self.config.fallback_file}")
        print(f"  Knowledge base enabled: {self.config.kb_enabled}")
        print(f"  Model: {self.config.model}")
        print(f"  Use simulation: {self.use_simulation}")
        
        # Create timestamped output directory
        output_dir, subtask_problem_dir = create_output_directory(self.config.output_dir)
        
        # Step 1: Generate subtasks
        print("\n=== STEP 1: Generating subtasks ===")
        start_time = time.time()
        
        try:
            subtasks, pddl_goals = self.task_generator.generate_subtasks(self.config.task_instruction)
        except Exception as e:
            print(f"Error generating subtasks: {e}")
            return
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Save generated subtasks
        subtasks_output_file = output_dir / "subtasks_with_pddl_goals.txt"
        self.task_generator.save_generated_subtasks(subtasks, pddl_goals, subtasks_output_file)
        print(f"Generation time: {elapsed_time:.2f} seconds")
        
        if not pddl_goals:
            print("No PDDL goals found. Exiting.")
            return
        
        print(f"Found {len(subtasks)} subtasks and {len(pddl_goals)} PDDL goals")
        for i, (subtask, goal) in enumerate(zip(subtasks, pddl_goals)):
            print(f"Subtask {i+1}: {subtask}")
            print(f"PDDL Goal {i+1}: {goal}")
            print()
        
        # Step 2: Solve all subtasks and collect complete action plan
        print("\n=== STEP 2: Solving all subtasks ===")
        current_init_state = None
        all_action_plans = []
        all_robot_init_locations = []
        plan_sources = []
        complete_action_plan = []  # Combined action plan for simulation
        
        for i in range(len(subtasks)):
            subtask_num = i + 1
            subtask = subtasks[i]
            pddl_goal = pddl_goals[i]
            
            action_plan, updated_init_state, success, robot_init_location, plan_source = self.solve_subtask(
                subtask_num, subtask, pddl_goal, subtask_problem_dir, current_init_state
            )
            
            if success:
                all_action_plans.append(action_plan)
                all_robot_init_locations.append(robot_init_location)
                plan_sources.append(plan_source)
                current_init_state = updated_init_state
                
                # Add actions to complete plan
                complete_action_plan.extend(action_plan)
            else:
                print(f"Failed to solve subtask {subtask_num}. Stopping execution.")
                break
        
        # Step 3: Execute complete action plan in simulation
        if self.use_simulation and complete_action_plan:
            print(f"\n{'='*60}")
            print("STEP 3: EXECUTING COMPLETE ACTION PLAN IN SIMULATION")
            print(f"{'='*60}")
            print(f"Total actions to execute: {len(complete_action_plan)}")
            
            # Display complete action plan
            print("\nComplete Action Plan:")
            for i, action in enumerate(complete_action_plan, 1):
                print(f"  {i:2d}. {action}")
            
            print(f"\n--- Initializing Simulator Executor V4 ---")
            try:
                # Create executor (it handles its own simulation initialization)
                self.executor = SimulatorExecutor()
                
                print(f"\n--- Executing {len(complete_action_plan)} actions in simulation ---")
                execution_success = self.executor.execute_action_plan(complete_action_plan, verbose=True)
                
                if execution_success:
                    print(f"\n🎉 SUCCESS: All {len(complete_action_plan)} actions executed successfully!")
                else:
                    print(f"\n❌ FAILURE: Simulation execution failed")
                    
            except Exception as e:
                print(f"❌ Error during simulation: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 4: Generate summary and save results
        self._generate_summary(subtasks, pddl_goals, all_action_plans, all_robot_init_locations, 
                             plan_sources, output_dir, elapsed_time, complete_action_plan)
        
        # Save knowledge base if enabled (save to base output directory, not timestamped)
        if self.knowledge_base:
            kb_file = self.knowledge_base.save(self.config.output_dir)
        
        print(f"\n🏁 Plan reuse system completed!")
        print(f"📁 All output files saved in: {output_dir}")
    
    def _generate_summary(self, subtasks: List[str], pddl_goals: List[str], all_action_plans: List[List[str]], 
                         all_robot_init_locations: List[str], plan_sources: List[str], 
                         output_dir: Path, elapsed_time: float, complete_action_plan: List[str]) -> None:
        """Generate and save execution summary."""
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        print(f"Task: {self.config.task_instruction}")
        print(f"Total subtasks: {len(subtasks)}")
        print(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
        print(f"Knowledge base retrievals: {self.kb_retrievals}")
        print(f"Classical planner calls: {self.classical_calls}")
        print(f"Total actions in complete plan: {len(complete_action_plan)}")
        
        if self.knowledge_base:
            kb_stats = self.knowledge_base.get_stats()
            print(f"Knowledge base entries: {kb_stats['total_entries']}")
        
        summary_content = []
        summary_content.append(f"Task: {self.config.task_instruction}")
        summary_content.append(f"Total subtasks: {len(subtasks)}")
        summary_content.append(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
        summary_content.append(f"Knowledge base retrievals: {self.kb_retrievals}")
        summary_content.append(f"Classical planner calls: {self.classical_calls}")
        summary_content.append(f"Total actions in complete plan: {len(complete_action_plan)}")
        if self.knowledge_base:
            kb_stats = self.knowledge_base.get_stats()
            summary_content.append(f"Knowledge base entries: {kb_stats['total_entries']}")
        summary_content.append("")
        
        # Individual subtask details
        for i, (action_plan, robot_init_location, plan_source) in enumerate(zip(all_action_plans, all_robot_init_locations, plan_sources)):
            subtask_display = f"Subtask-{i+1}: {subtasks[i]}"
            robot_init_display = f"robot_init: {robot_init_location if robot_init_location else 'Unknown'}"
            goal_display = f"Goal-{i+1} (pddl): {pddl_goals[i]}"
            source_display = f"Plan source: {plan_source}"
            actions_display = f"Actions ({len(action_plan)}):"
            
            print(f"\n{subtask_display}")
            print(f"{robot_init_display}")
            print(f"{goal_display}")
            print(f"{source_display}")
            print(f"{actions_display}")
            
            summary_content.extend([subtask_display, robot_init_display, goal_display, source_display, actions_display])
            
            for j, action in enumerate(action_plan, 1):
                action_line = f"  {j}. {action}"
                print(action_line)
                summary_content.append(action_line)
            
            summary_content.append("")
        
        # Complete action plan
        summary_content.append("=" * 50)
        summary_content.append("COMPLETE ACTION PLAN FOR SIMULATION")
        summary_content.append("=" * 50)
        for i, action in enumerate(complete_action_plan, 1):
            action_line = f"{i:2d}. {action}"
            summary_content.append(action_line)
        
        # Save summary to file
        summary_file = output_dir / "execution_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_content))
        print(f"Execution summary saved to: {summary_file}")
        
        # Save complete action plan separately
        action_plan_file = output_dir / "complete_action_plan.txt"
        with open(action_plan_file, 'w') as f:
            f.write("# Complete Action Plan for Simulation Execution\n")
            f.write(f"# Generated from task: {self.config.task_instruction}\n")
            f.write(f"# Total actions: {len(complete_action_plan)}\n\n")
            for i, action in enumerate(complete_action_plan, 1):
                f.write(f"{i:2d}. {action}\n")
        print(f"Complete action plan saved to: {action_plan_file}")
        
        # Save run configuration
        config_file = output_dir / "run_config.txt"
        with open(config_file, 'w') as f:
            f.write(f"Run Configuration\n")
            f.write(f"================\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task: {self.config.task_instruction}\n")
            f.write(f"Use LLM: {self.config.use_llm}\n")
            f.write(f"Fallback file: {self.config.fallback_file}\n")
            f.write(f"Model: {self.config.model}\n")
            f.write(f"Max tokens: {self.config.max_tokens}\n")
            f.write(f"Generation time: {elapsed_time:.2f} seconds\n")
            f.write(f"Knowledge base enabled: {self.config.kb_enabled}\n")
            f.write(f"Use simulation: {self.use_simulation}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Domain file: {self.config.domain_file}\n")
            f.write(f"Problem file: {self.config.problem_file}\n")
            f.write(f"Total actions executed: {len(complete_action_plan)}\n")
            f.write(f"Knowledge base retrievals: {self.kb_retrievals}\n")
            f.write(f"Classical planner calls: {self.classical_calls}\n")
        print(f"Run configuration saved to: {config_file}")


def main():
    """Main entry point."""
    try:
        # Load configuration
        # config = PlanReuseConfig(use_llm=False, fallback_file=Path("config_files/sample_falback.txt"))
        config = PlanReuseConfig()
        config.validate()
        
        system = PlanReuseSystemV3(config, use_simulation=True)
        # system = PlanReuseSystemV3(config, use_simulation=False)
        system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and file paths.")


if __name__ == "__main__":
    main()