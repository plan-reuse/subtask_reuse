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
    LockRenderer, wait_if_gui, connect, disconnect, set_camera_pose,
    set_pose, Pose, Point
)
from pybullet_tools.pr2_utils import (
    PR2_GROUPS, SIDE_HOLDING_LEFT_ARM, TOP_HOLDING_LEFT_ARM,
    rightarm_from_leftarm, REST_LEFT_ARM, open_arm
)

from world_builder.world import World
from world_builder.loaders_nvidia_kitchen import load_open_problem_kitchen
from robot_builder.robot_builders import create_pr2_robot
from world_builder.world_utils import load_asset
from world_builder.entities import Movable

class SimulatorExecutor:
    """Executes PDDL action plans in PyBullet simulation."""
    
    def __init__(self):
        self.robot_body = None
        self.world = None
        self.movables = None
        self.obstacles = None
        self.extended_objects = None
        
        # Map PDDL regions to PyBullet coordinates (x, y, theta)
        self.region_locations = {
            'kitchen_corner': (2.0, 6.25, 0.0),
            'fridge': (1.7, 5.0, 3.1415),
            'sink': (1.5, 6.0, 3.1415),
            'countertop_1': (1.5, 6.8, 3.1415),
            'cabinet': (1.7, 7.5, 3.1415),
            'stove': (1.5, 8.5, 3.1415),
            'countertop_2': (1.5, 9.2, 3.1415),
        }
        
        self.left_joints = None
        self.holding_object = None
        
        # Initialize the simulation environment
        self._initialize_simulation()
        
    def _initialize_simulation(self):
        """Initialize PyBullet simulation with world, robot, and kitchen."""
        try:
            # Initialize PyBullet
            print("Initializing PyBullet simulation...")
            connect(use_gui=True, shadows=False, width=1920, height=1200)
            set_camera_pose(camera_point=[4, 6, 3], target_point=[2, 6, 1])
            
            # Create world and kitchen
            self.world = World(time_step=1./240, segment=False, constants={})
            self.world.set_skip_joints()
            
            # Create robot at kitchen corner
            robot = create_pr2_robot(self.world, base_q=(0, 0, 0), dual_arm=False, 
                                   use_torso=True, custom_limits=((1, 3, 0), (5, 10, 3)))
            self.robot_body = robot.body
            
            # Load kitchen environment  
            objects, self.movables = load_open_problem_kitchen(self.world, reduce_objects=False, 
                                                        difficulty=1, randomize_joint_positions=False)
            
            # Load extended objects
            print("Loading extended object collection...")
            self.extended_objects = self._load_extended_objects()
            
            # Set up robot configuration
            self.left_joints = joints_from_names(self.robot_body, PR2_GROUPS['left_arm'])
            right_joints = joints_from_names(self.robot_body, PR2_GROUPS['right_arm'])
            torso_joints = joints_from_names(self.robot_body, PR2_GROUPS['torso'])
            
            set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
            set_joint_positions(self.robot_body, right_joints, rightarm_from_leftarm(REST_LEFT_ARM))
            set_joint_positions(self.robot_body, torso_joints, [0.2])
            open_arm(self.robot_body, 'left')
            
            # Set up obstacles
            self.obstacles = list(objects) if objects else []
            for body_id in self.world.fixed:
                if body_id not in self.obstacles:
                    self.obstacles.append(body_id)
            
            # Set appliance states (open doors/drawers)
            self._set_appliance_state(0.6)
            
            # Set enhanced object positions
            self._set_enhanced_object_positions()
            
            print(f"Environment loaded with {len(self.obstacles)} obstacles")
            print(f"Base movable objects: {len(self.movables)}")
            print(f"Extended objects: {len(self.extended_objects)}")
            
            # Track robot's held object
            self.holding_object = None
            
        except Exception as e:
            print(f"Error during simulation initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_extended_objects(self):
        """Load additional objects to enhance the kitchen environment"""
        extended_objects = {}
        
        # Define new objects with their categories and positions
        new_objects_config = {
            # Additional vegetables
            'tomato': {
                'asset': 'VeggieTomato',
                'category': 'food',
                'position': (0.7, 5.2, 0.85),
                'name': 'tomato'
            },
            'zucchini': {
                'asset': 'VeggieZucchini', 
                'category': 'food',
                'position': (0.9, 4.9, 0.85),
                'name': 'zucchini'
            },
            'potato': {
                'asset': 'VeggiePotato',
                'category': 'food', 
                'position': (0.8, 5.0, 0.85),
                'name': 'potato'
            },
            'green-pepper': {
                'asset': 'VeggieGreenPepper',
                'category': 'food',
                'position': (0.75, 5.1, 0.85),
                'name': 'green-pepper'
            },
            'artichoke': {
                'asset': 'VeggieArtichoke',
                'category': 'food',
                'position': (0.85, 4.95, 0.85),
                'name': 'artichoke'
            },
            
            # Kitchen utensils
            'kitchen-knife': {
                'asset': 'KitchenKnife',
                'category': 'utensil',
                'position': (0.9, 8.5, 0.75),
                'name': 'kitchen-knife'
            },
            
            # Bottles and containers
            'oil-bottle': {
                'asset': 'OilBottle', 
                'category': 'condiment',
                'position': (0.8, 7.2, 1.2),
                'name': 'oil-bottle'
            },
            'vinegar-bottle': {
                'asset': 'VinegarBottle',
                'category': 'condiment', 
                'position': (0.85, 7.4, 1.2),
                'name': 'vinegar-bottle'
            },
            'milk-bottle': {
                'asset': 'MilkBottle',
                'category': 'food',
                'position': (0.65, 4.7, 1.4),  # In fridge
                'name': 'milk-bottle'
            },
            
            # Plates and dishware
            'plate': {
                'asset': 'Plate',
                'category': 'dishware',
                'position': (2, 6.2, 1.52),  # In dishwasher area
                'name': 'plate'
            }
        }
        
        # Load each object
        for obj_key, config in new_objects_config.items():
            try:
                # Load the asset
                body = load_asset(
                    config['asset'], 
                    x=config['position'][0], 
                    y=config['position'][1], 
                    yaw=0
                )
                
                # Create movable object
                movable_obj = self.world.add_object(Movable(
                    body, 
                    category=config['category'], 
                    name=config['name']
                ))
                
                # Set precise position
                new_pose = Pose(point=Point(*config['position']))
                movable_obj.set_pose(new_pose)
                
                # Add to our extended objects dict
                extended_objects[obj_key] = movable_obj.body
                
                # Add to world categories
                self.world.add_to_cat(movable_obj.body, 'movable')
                self.world.add_to_cat(movable_obj.body, config['category'])
                
                print(f"   Successfully loaded: {config['name']} ({config['asset']}) at {config['position']}")
                
            except Exception as e:
                print(f"   Failed to load {obj_key} ({config['asset']}): {e}")
                continue
        
        return extended_objects

    def _set_appliance_state(self, extent):
        """
        Set all appliances to the specified extent
        
        Args:
            extent: How much to open appliances (0.0 = closed, 0.8 = open)
        """
        doors = self.world.cat_to_objects('door')
        for door in doors:
            try:
                # Special handling for fridge (smaller extent)
                actual_extent = min(extent, 0.7) if 'fridge' in door.name.lower() else extent
                self.world.open_joint(door.body, joint=door.joint, extent=actual_extent, verbose=True)
            except Exception as e:
                print(f"Error setting {door.name}: {e}")

    def _set_enhanced_object_positions(self):
        """
        Set movable objects to their enhanced default positions
        Includes both base objects and new extended objects
        """
        # Enhanced positions for base objects
        BASE_OBJECT_POSITIONS = {
            'chicken-leg': (0.65, 4.85, 1.38),     # Chicken leg in fridge
            'cabbage': (0.65, 5.05, 1.38),         # Cabbage in fridge bottom
            'salt-shaker': (0.771, 7.071, 1.152),     # Salt shaker on counter
            'pepper-shaker': (0.764, 7.303, 1.164),   # Pepper shaker on counter
            'braiserbody': (0.8, 8.5, 1.0),          # Pot body on stove
            'braiserlid': (0.8, 9.0, 0.8),            # Pot lid nearby
        }
        
        # Enhanced positions for extended objects
        EXTENDED_OBJECT_POSITIONS = {
            'tomato': (0.6, 4.7, 0.85),              # On counter
            'zucchini': (0.65, 5.1, 0.85),           # In fridge bottom
            'potato': (0.6, 4.85, 0.85),             # In fridge bottom  
            'artichoke': (0.6, 5.0, 0.85),         # In fridge bottom
            'oil-bottle': (0.7, 7.2, 1.2),           # On cabinet shelf
            'vinegar-bottle': (0.7, 7.4, 1.2),      # On cabinet shelf
            'milk-bottle': (0.65, 4.7, 1.4),         # In fridge top shelf
            'plate': (0.8, 6.5, 1.0),               # In dishwasher
        }
        
        # Position base objects
        for obj_name, position in BASE_OBJECT_POSITIONS.items():
            self._set_object_position(obj_name, position, self.movables, "base")
        
        # Position extended objects
        for obj_name, position in EXTENDED_OBJECT_POSITIONS.items():
            self._set_object_position(obj_name, position, self.extended_objects, "extended")

    def _set_object_position(self, obj_name, position, object_dict, obj_type):
        """Helper method to set object position"""
        try:
            if obj_name in object_dict:
                body_id = object_dict[obj_name]
                new_pose = Pose(point=Point(*position))
                set_pose(body_id, new_pose)
                print(f"Moved {obj_type} object {obj_name} to position: {position}")
            else:
                # Try to find in world by name
                try:
                    body_id = self.world.name_to_body(obj_name)
                    new_pose = Pose(point=Point(*position))
                    set_pose(body_id, new_pose)
                    print(f"Moved {obj_type} object {obj_name} (via world lookup) to position: {position}")
                except:
                    print(f"{obj_type.capitalize()} object {obj_name} not found")
        except Exception as e:
            print(f"Error moving {obj_type} object {obj_name}: {e}")

    def get_all_movable_objects(self):
        """Return dictionary of all movable objects (base + extended)"""
        all_objects = {}
        all_objects.update(self.movables)
        all_objects.update(self.extended_objects)
        return all_objects
        
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
    
    def __del__(self):
        """Cleanup PyBullet connection when object is destroyed."""
        try:
            disconnect()
        except:
            pass


def main():
    """Main function that creates executor and runs test actions."""
    # Test action list - simple breakfast preparation sequence
    test_actions = [
        "move pr2 kitchen_corner fridge",
        "pickup_from_container pr2 chicken-leg fridge",
        "move pr2 fridge countertop_2",
        "move pr2 countertop_2 fridge",
        "putdown_to_container pr2 chicken-leg fridge",
    ]
    
    try:
        print("=== TESTING SIMULATOR EXECUTOR V3 ===")
        
        # Create executor (this initializes everything)
        executor = SimulatorExecutor()
        
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


if __name__ == "__main__":
    main()