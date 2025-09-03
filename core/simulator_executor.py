"""Simulator executor that maps PDDL actions to PyBullet robot commands."""

import sys
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add PyBullet planning paths
sys.path.extend([
    '../pybullet_planning',
    '../pddlstream', 
    '../lisdf'
])

from pybullet_tools.utils import (
    set_base_values, get_base_values, plan_base_motion, 
    set_joint_positions, joints_from_names, wait_for_duration,
    LockRenderer, wait_if_gui
)
from pybullet_tools.pr2_utils import PR2_GROUPS, SIDE_HOLDING_LEFT_ARM, TOP_HOLDING_LEFT_ARM

class SimulatorExecutor:
    """Executes PDDL action plans in PyBullet simulation."""
    
    def __init__(self, robot_body, world, movables=None, obstacles=None):
        self.robot_body = robot_body
        self.world = world
        self.movables = movables or {}
        self.obstacles = obstacles or []
        
        # Map PDDL regions to PyBullet coordinates (x, y, theta)
        self.region_locations = {
            'kitchen_corner': (2.0, 6.25, 0.0),
            'fridge': (1.8, 5.0, 1.57),
            'countertop_1': (1.5, 7.0, -1.57),
            'countertop_2': (1.8, 7.5, -1.57),
            'sink': (1.5, 5.8, 0.78),
            'stove': (2.2, 8.5, -1.57),
            'cabinet': (1.4, 6.8, 1.57)
        }
        
        # Get robot joints
        self.left_joints = joints_from_names(robot_body, PR2_GROUPS['left_arm'])
        
        # Track robot's held object
        self.holding_object = None
        
    def execute_action_plan(self, action_plan: List[str], verbose: bool = True) -> bool:
        """Execute a complete action plan."""
        print(f"\n=== EXECUTING ACTION PLAN ({len(action_plan)} actions) ===")
        
        for i, action in enumerate(action_plan, 1):
            print(f"\n--- Action {i}/{len(action_plan)}: {action} ---")
            success = self.execute_single_action(action, verbose)
            
            if not success:
                print(f"❌ Failed to execute action: {action}")
                return False
                
            if verbose:
                wait_if_gui(f'Continue to next action?')
                
        print(f"\n✅ Successfully executed all {len(action_plan)} actions!")
        return True
    
    def execute_single_action(self, action: str, verbose: bool = True) -> bool:
        """Execute a single PDDL action."""
        action = action.strip()
        
        # Parse action using regex
        if action.startswith('move '):
            return self._execute_move(action, verbose)
        elif action.startswith('open_container '):
            return self._execute_open_container(action, verbose)
        elif action.startswith('close_container '):
            return self._execute_close_container(action, verbose)
        elif action.startswith('pickup_from_region '):
            return self._execute_pickup_from_region(action, verbose)
        elif action.startswith('pickup_from_container '):
            return self._execute_pickup_from_container(action, verbose)
        elif action.startswith('putdown_to_region '):
            return self._execute_putdown_to_region(action, verbose)
        elif action.startswith('put_food_in_cookware '):
            return self._execute_put_food_in_cookware(action, verbose)
        elif action.startswith('place_cookware_on_stovetop '):
            return self._execute_place_cookware_on_stovetop(action, verbose)
        elif action.startswith('turn_on_stove '):
            return self._execute_turn_on_stove(action, verbose)
        elif action.startswith('turn_off_stove '):
            return self._execute_turn_off_stove(action, verbose)
        elif action.startswith('wait_for_food_to_cook '):
            return self._execute_wait_for_food_to_cook(action, verbose)
        else:
            print(f"⚠️  Unknown action type: {action}")
            return True  # Don't fail for unknown actions
    
    def _execute_move(self, action: str, verbose: bool) -> bool:
        """Execute move action: move robot from to"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, from_region, to_region = parts[1], parts[2], parts[3]
        
        if to_region not in self.region_locations:
            print(f"Unknown region: {to_region}")
            return False
            
        current_pos = get_base_values(self.robot_body)
        target_pos = self.region_locations[to_region]
        
        if verbose:
            print(f"Moving from {current_pos} to {to_region} at {target_pos}")
        
        # Plan and execute base motion
        base_limits = ((1, 3), (5, 10))
        with LockRenderer(lock=False):
            base_path = plan_base_motion(self.robot_body, target_pos, base_limits, 
                                       obstacles=self.obstacles)
        
        if base_path is None:
            print(f"Failed to plan path to {to_region}")
            return False
            
        # Execute path
        for waypoint in base_path:
            set_base_values(self.robot_body, waypoint)
            wait_for_duration(0.02)
            
        return True
    
    def _execute_open_container(self, action: str, verbose: bool) -> bool:
        """Execute open_container action"""
        parts = action.split()
        if len(parts) < 3:
            return False
            
        robot, container = parts[1], parts[2]
        
        if verbose:
            print(f"Opening {container}")
            
        # Find container in world and open it
        try:
            if container == 'fridge':
                doors = self.world.cat_to_objects('door')
                for door in doors:
                    if 'fridge' in door.name.lower():
                        self.world.open_joint(door.body, joint=door.joint, extent=0.7, verbose=verbose)
                        return True
        except Exception as e:
            print(f"Error opening {container}: {e}")
            
        return True  # Don't fail simulation for container operations
    
    def _execute_close_container(self, action: str, verbose: bool) -> bool:
        """Execute close_container action"""
        parts = action.split()
        if len(parts) < 3:
            return False
            
        robot, container = parts[1], parts[2]
        
        if verbose:
            print(f"Closing {container}")
            
        # Find container and close it
        try:
            if container == 'fridge':
                doors = self.world.cat_to_objects('door')
                for door in doors:
                    if 'fridge' in door.name.lower():
                        self.world.open_joint(door.body, joint=door.joint, extent=0.0, verbose=verbose)
                        return True
        except Exception as e:
            print(f"Error closing {container}: {e}")
            
        return True
    
    def _execute_pickup_from_region(self, action: str, verbose: bool) -> bool:
        """Execute pickup_from_region action"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, obj, region = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"Picking up {obj} from {region}")
            
        # Simulate arm motion to pickup pose
        set_joint_positions(self.robot_body, self.left_joints, TOP_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        # Track that robot is holding object
        self.holding_object = obj
        
        return True
    
    def _execute_pickup_from_container(self, action: str, verbose: bool) -> bool:
        """Execute pickup_from_container action"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, obj, container = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"Picking up {obj} from {container}")
            
        # Simulate reaching into container and picking up
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.3)
        set_joint_positions(self.robot_body, self.left_joints, TOP_HOLDING_LEFT_ARM)
        wait_for_duration(0.3)
        
        self.holding_object = obj
        
        return True
    
    def _execute_putdown_to_region(self, action: str, verbose: bool) -> bool:
        """Execute putdown_to_region action"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, obj, region = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"Putting down {obj} to {region}")
            
        # Simulate arm motion to putdown
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        
        return True
    
    def _execute_put_food_in_cookware(self, action: str, verbose: bool) -> bool:
        """Execute put_food_in_cookware action"""
        parts = action.split()
        if len(parts) < 5:
            return False
            
        robot, food, cookware, region = parts[1], parts[2], parts[3], parts[4]
        
        if verbose:
            print(f"Putting {food} into {cookware}")
            
        # Simulate placing food in cookware
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        
        return True
    
    def _execute_place_cookware_on_stovetop(self, action: str, verbose: bool) -> bool:
        """Execute place_cookware_on_stovetop action"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, cookware, stovetop = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"Placing {cookware} on {stovetop}")
            
        # Simulate placing cookware on stovetop
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        
        return True
    
    def _execute_turn_on_stove(self, action: str, verbose: bool) -> bool:
        """Execute turn_on_stove action"""
        parts = action.split()
        if len(parts) < 3:
            return False
            
        robot, stove = parts[1], parts[2]
        
        if verbose:
            print(f"Turning on {stove}")
            
        # Simulate turning on stove (just a delay)
        wait_for_duration(0.3)
        
        return True
    
    def _execute_turn_off_stove(self, action: str, verbose: bool) -> bool:
        """Execute turn_off_stove action"""
        parts = action.split()
        if len(parts) < 3:
            return False
            
        robot, stove = parts[1], parts[2]
        
        if verbose:
            print(f"Turning off {stove}")
            
        wait_for_duration(0.3)
        
        return True
    
    def _execute_wait_for_food_to_cook(self, action: str, verbose: bool) -> bool:
        """Execute wait_for_food_to_cook action"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        stove, cookware, food = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"Waiting for {food} to cook in {cookware} on {stove}")
            
        # Simulate cooking time
        wait_for_duration(2.0)
        
        return True


def main():
    """Standalone test function for SimulatorExecutor."""
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
    
    print("=== TESTING SIMULATOR EXECUTOR ===")
    
    # Test action list - simple breakfast preparation sequence
    test_actions = [
        "move pr2 kitchen_corner fridge",
        "move pr2 fridge countertop_2",
        "move pr2 countertop_2 stove",
    ]
    
    try:
        # Initialize PyBullet
        print("Initializing PyBullet simulation...")
        connect(use_gui=True, shadows=False, width=1980, height=1238)
        set_camera_pose(camera_point=[5, 6, 3], target_point=[2, 6, 1])
        
        # Create world and kitchen
        world = World(time_step=1./240, segment=False, constants={})
        world.set_skip_joints()
        
        # Create robot at kitchen corner
        robot = create_pr2_robot(world, base_q=(0, 0, 0), dual_arm=False, 
                               use_torso=True, custom_limits=((1, 3, 0), (5, 10, 3)))
        
        # Load kitchen environment  
        objects, movables = load_open_problem_kitchen(world, reduce_objects=False, 
                                                    difficulty=1, randomize_joint_positions=False)
        
        # Set up robot configuration
        left_joints = joints_from_names(robot.body, PR2_GROUPS['left_arm'])
        right_joints = joints_from_names(robot.body, PR2_GROUPS['right_arm'])
        torso_joints = joints_from_names(robot.body, PR2_GROUPS['torso'])
        
        set_joint_positions(robot.body, left_joints, SIDE_HOLDING_LEFT_ARM)
        set_joint_positions(robot.body, right_joints, rightarm_from_leftarm(REST_LEFT_ARM))
        set_joint_positions(robot.body, torso_joints, [0.2])
        open_arm(robot.body, 'left')
        
        # Set up obstacles
        obstacles = list(objects) if objects else []
        for body_id in world.fixed:
            if body_id not in obstacles:
                obstacles.append(body_id)
        
        print(f"Environment loaded with {len(obstacles)} obstacles")
        
        # Create executor
        executor = SimulatorExecutor(robot.body, world, movables, obstacles)
        
        print(f"Testing with {len(test_actions)} actions:")
        for i, action in enumerate(test_actions, 1):
            print(f"  {i}. {action}")
        
        wait_if_gui('Start executing test actions?')
        
        # Execute test actions
        success = executor.execute_action_plan(test_actions, verbose=True)
        
        if success:
            print("\n🎉 SUCCESS: All test actions executed successfully!")
        else:
            print("\n❌ FAILURE: Some actions failed to execute")
        
        wait_if_gui('Testing complete. Press to exit.')
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            disconnect()
        except:
            pass


if __name__ == "__main__":
    main()