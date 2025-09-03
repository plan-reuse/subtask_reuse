"""Unified main entry point for the plan reuse system with PyBullet simulation integration."""

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
from core.abstraction import PlanAbstractor
from core.simulator_executor import SimulatorExecutor
from utils.file_ops import create_output_directory
from utils.parsing import create_problem_file_with_goal, create_updated_problem_file, extract_robot_location

# PyBullet imports
from pybullet_tools.utils import (
    connect, disconnect, set_camera_pose, set_base_values, 
    joints_from_names, set_joint_positions, wait_if_gui
)
from pybullet_tools.pr2_utils import (
    PR2_GROUPS, SIDE_HOLDING_LEFT_ARM, rightarm_from_leftarm, 
    REST_LEFT_ARM, open_arm
)

from world_builder.world import World
from world_builder.loaders_nvidia_kitchen import load_open_problem_kitchen
from robot_builder.robot_builders import create_pr2_robot


class PlanReuseSystemWithSimulation:
    """Main system orchestrating plan reuse functionality with PyBullet simulation."""
    
    def __init__(self, config: PlanReuseConfig, use_simulation: bool = True):
        self.config = config
        self.use_simulation = use_simulation
        
        # Initialize components
        self.task_generator = TaskGenerator(config)
        self.knowledge_base = KnowledgeBase(config) if config.kb_enabled else None
        self.planner = PDDLPlanner(config)
        self.state_manager = StateManager()
        self.abstractor = PlanAbstractor()
        
        # Simulation components
        self.world = None
        self.robot = None
        self.executor = None
        self.movables = {}
        self.obstacles = []
        
        # Statistics
        self.kb_retrievals = 0
        self.classical_calls = 0
    
    def initialize_simulation(self) -> bool:
        """Initialize PyBullet simulation environment."""
        if not self.use_simulation:
            return True
            
        try:
            print("\n=== INITIALIZING SIMULATION ===")
            
            # Connect to PyBullet
            connect(use_gui=True, shadows=False, width=1980, height=1238)
            set_camera_pose(camera_point=[5, 6, 3], target_point=[2, 6, 1])
            
            # Create world and kitchen
            self.world = World(time_step=1./240, segment=False, constants={})
            self.world.set_skip_joints()
            
            # Create robot
            self.robot = create_pr2_robot(self.world, base_q=(2, 6.25, 0), dual_arm=False, 
                                        use_torso=True, custom_limits=((1, 3, 0), (5, 10, 3)))
            
            # Load kitchen environment
            objects, self.movables = load_open_problem_kitchen(self.world, reduce_objects=False, 
                                                             difficulty=1, randomize_joint_positions=False)
            
            # Set up robot configuration
            left_joints = joints_from_names(self.robot.body, PR2_GROUPS['left_arm'])
            right_joints = joints_from_names(self.robot.body, PR2_GROUPS['right_arm'])
            torso_joints = joints_from_names(self.robot.body, PR2_GROUPS['torso'])
            
            set_joint_positions(self.robot.body, left_joints, SIDE_HOLDING_LEFT_ARM)
            set_joint_positions(self.robot.body, right_joints, rightarm_from_leftarm(REST_LEFT_ARM))
            set_joint_positions(self.robot.body, torso_joints, [0.2])
            open_arm(self.robot.body, 'left')
            
            # Set up obstacles for motion planning
            self.obstacles = list(objects) if objects else []
            
            # Add kitchen fixtures as obstacles
            for body_id in self.world.fixed:
                if body_id not in self.obstacles:
                    self.obstacles.append(body_id)
            
            # Create executor
            self.executor = SimulatorExecutor(self.robot.body, self.world, 
                                            self.movables, self.obstacles)
            
            print(f"✅ Simulation initialized with {len(self.obstacles)} obstacles")
            
            wait_if_gui('Start task execution?')
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize simulation: {e}")
            return False
    
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
            
            # Execute in simulation if enabled
            if self.use_simulation and self.executor:
                print(f"\n--- Executing Subtask {subtask_num} in Simulation ---")
                execution_success = self.executor.execute_action_plan(action_plan, verbose=True)
                
                if not execution_success:
                    print(f"⚠️  Simulation execution failed for subtask {subtask_num}")
                else:
                    print(f"✅ Simulation execution successful for subtask {subtask_num}")
            
            # Update initial state
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
        """Run the complete plan reuse system with simulation."""
        print("="*60)
        print("PLAN REUSE SYSTEM WITH SIMULATION")
        print("="*60)
        print(f"Configuration:")
        print(f"  Task: {self.config.task_instruction}")
        print(f"  Use LLM: {self.config.use_llm}")
        print(f"  Fallback file: {self.config.fallback_file}")
        print(f"  Knowledge base enabled: {self.config.kb_enabled}")
        print(f"  Model: {self.config.model}")
        print(f"  Use simulation: {self.use_simulation}")
        
        # Initialize simulation if enabled
        if self.use_simulation:
            if not self.initialize_simulation():
                print("Failed to initialize simulation. Exiting.")
                return
        
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
        
        # Step 2: Solve all subtasks
        print("\n=== STEP 2: Solving subtasks ===")
        current_init_state = None
        all_action_plans = []
        all_robot_init_locations = []
        plan_sources = []
        
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
            else:
                print(f"Failed to solve subtask {subtask_num}. Stopping execution.")
                break
        
        # Step 3: Generate summary and save results
        self._generate_summary(subtasks, pddl_goals, all_action_plans, all_robot_init_locations, 
                             plan_sources, output_dir, elapsed_time)
        
        # Save knowledge base if enabled
        if self.knowledge_base:
            kb_file = self.knowledge_base.save(output_dir)
        
        # Keep simulation running for inspection
        if self.use_simulation:
            wait_if_gui('Finished! Press to close simulation.')
            disconnect()
    
    def _generate_summary(self, subtasks: List[str], pddl_goals: List[str], all_action_plans: List[List[str]], 
                         all_robot_init_locations: List[str], plan_sources: List[str], 
                         output_dir: Path, elapsed_time: float) -> None:
        """Generate and save execution summary."""
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        print(f"Task: {self.config.task_instruction}")
        print(f"Total subtasks: {len(subtasks)}")
        print(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
        print(f"Knowledge base retrievals: {self.kb_retrievals}")
        print(f"Classical planner calls: {self.classical_calls}")
        
        if self.knowledge_base:
            kb_stats = self.knowledge_base.get_stats()
            print(f"Knowledge base entries: {kb_stats['total_entries']}")
        
        total_actions = 0
        summary_content = []
        summary_content.append(f"Task: {self.config.task_instruction}")
        summary_content.append(f"Total subtasks: {len(subtasks)}")
        summary_content.append(f"Subtasks solved: {len(all_action_plans)}/{len(subtasks)}")
        summary_content.append(f"Knowledge base retrievals: {self.kb_retrievals}")
        summary_content.append(f"Classical planner calls: {self.classical_calls}")
        if self.knowledge_base:
            kb_stats = self.knowledge_base.get_stats()
            summary_content.append(f"Knowledge base entries: {kb_stats['total_entries']}")
        summary_content.append("")
        
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
            total_actions += len(action_plan)
        
        print(f"\nTotal actions executed: {total_actions}")
        
        # Save summary to file
        summary_file = output_dir / "execution_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_content))
        print(f"Execution summary saved to: {summary_file}")
        
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
            f.write(f"Total actions executed: {total_actions}\n")
            f.write(f"Knowledge base retrievals: {self.kb_retrievals}\n")
            f.write(f"Classical planner calls: {self.classical_calls}\n")
        print(f"Run configuration saved to: {config_file}")
        
        print(f"\nAll output files saved in: {output_dir}")


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = PlanReuseConfig(use_llm=False, fallback_file=Path("config_files/sample_falback.txt"))
        config.validate()
        
        # Create and run the system with simulation
        system = PlanReuseSystemWithSimulation(config, use_simulation=True)
        system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and file paths.")


if __name__ == "__main__":
    main()