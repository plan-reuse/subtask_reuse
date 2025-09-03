"""Simulator executor that maps PDDL actions to PyBullet robot commands with PDDL name mapping."""

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
    set_pose, Pose, Point, get_pose, add_fixed_constraint, remove_fixed_constraint,
    link_from_name, get_com_pose, multiply, invert, unit_point, unit_quat, get_link_pose,
    body_from_end_effector
)
from pybullet_tools.pr2_utils import (
    PR2_GROUPS, SIDE_HOLDING_LEFT_ARM, TOP_HOLDING_LEFT_ARM,
    rightarm_from_leftarm, REST_LEFT_ARM, open_arm, LEFT_ARM, get_gripper_link
)
import pybullet as p

from world_builder.world import World
from world_builder.loaders_nvidia_kitchen import load_open_problem_kitchen
from robot_builder.robot_builders import create_pr2_robot
from world_builder.world_utils import load_asset
from world_builder.entities import Movable

class SimulatorExecutor:
    """Executes PDDL action plans in PyBullet simulation with PDDL name mapping."""
    
    def __init__(self):
        self.robot_body = None
        self.world = None
        self.movables = None
        self.obstacles = None
        self.extended_objects = None
        
        # PDDL name to simulator name mapping
        self.pddl_to_sim_names = {
            # Cookware mapping
            'pot': 'braiserbody',
            
            # Food mapping  
            'chicken': 'chicken-leg',
            'salt': 'salt-shaker',
            'pepper': 'pepper-shaker',
            'oil': 'oil-bottle',
            'vinegar': 'vinegar-bottle',
            'milk': 'milk-bottle',
            
            # Objects that don't need mapping (same names)
            'plate': 'plate',
            'potato': 'potato',
            'tomato': 'tomato',
            'cabbage': 'cabbage',
            'zucchini': 'zucchini',
            'artichoke': 'artichoke',
            
            # Regions and containers (no mapping needed)
            'pr2': 'pr2',
            'kitchen_corner': 'kitchen_corner',
            'countertop_1': 'countertop_1',
            'countertop_2': 'countertop_2',
            'sink': 'sink',
            'fridge': 'fridge',
            'cabinet': 'cabinet',
            'stove': 'stove'
        }
        
        # Reverse mapping for debugging
        self.sim_to_pddl_names = {v: k for k, v in self.pddl_to_sim_names.items()}
        
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
        
        # Container positions for object placement
        self.container_positions = {
            'fridge': {
                'bottom_shelf': (0.65, 4.85, 1.1),
                'top_shelf': (0.65, 4.7, 1.4)
            },
            'cabinet': {
                'shelf': (0.7, 7.3, 1.2)
            }
        }
        
        # Region positions for object placement
        self.region_object_positions = {
            'kitchen_corner': (2.0, 6.25, 0.8),
            'countertop_1': (1.3, 6.8, 1.0),
            'countertop_2': (1.3, 9.2, 1.0),
            'sink': (1.3, 6.0, 1.0),
            'fridge': (0.65, 4.85, 1.1),     # Inside fridge
            'cabinet': (0.7, 7.3, 1.2),      # On cabinet shelf
            'stove': (1.3, 8.5, 1.0)         # On stove surface
        }
        
        # Cookware and utensil relative positions
        self.cookware_food_offset = (0.0, 0.0, 0.05)  # Food inside cookware
        self.utensil_food_offset = (0.0, 0.0, 0.02)   # Food on utensil
        self.stovetop_cookware_position = (0.75, 8.0, 1.0)  # Cookware on stove
        
        self.left_joints = None
        self.holding_object = None  # Using PDDL names
        self.attached_object_id = None  # ID of attached object
        self.grasp_pose = None  # Relative transform between gripper and object
        self.gripper_link = None  # PR2 gripper link
        
        # Initialize the simulation environment
        self._initialize_simulation()
        
    def _map_pddl_to_sim_name(self, pddl_name: str) -> str:
        """Convert PDDL object name to simulator object name."""
        sim_name = self.pddl_to_sim_names.get(pddl_name, pddl_name)
        if pddl_name != sim_name:
            print(f"    Name mapping: {pddl_name} → {sim_name}")
        return sim_name
        
    def _map_sim_to_pddl_name(self, sim_name: str) -> str:
        """Convert simulator object name to PDDL object name."""
        return self.sim_to_pddl_names.get(sim_name, sim_name)
        
    def _initialize_simulation(self):
        """Initialize PyBullet simulation with world, robot, and kitchen."""
        try:
            # Initialize PyBullet
            print("Initializing PyBullet simulation...")
            connect(use_gui=True, shadows=False, width=1920, height=1200)
            set_camera_pose(camera_point=[3, 6, 4], target_point=[2, 6, 1])
            
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
            
            # Get gripper link for attachment
            self.gripper_link = get_gripper_link(self.robot_body, LEFT_ARM)
            
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
            
            # Print name mappings for verification
            print(f"\n=== PDDL Name Mappings ===")
            for pddl_name, sim_name in self.pddl_to_sim_names.items():
                if pddl_name != sim_name:
                    print(f"  {pddl_name} → {sim_name}")
            
            # Track robot's held object
            self.holding_object = None
            self.attached_object_id = None
            self.grasp_pose = None
            
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

    def _get_object_body_id(self, pddl_obj_name):
        """Helper to get body ID for an object using PDDL name, converting to simulator name if needed"""
        # Convert PDDL name to simulator name
        sim_obj_name = self._map_pddl_to_sim_name(pddl_obj_name)
        
        if sim_obj_name in self.movables:
            return self.movables[sim_obj_name]
        elif sim_obj_name in self.extended_objects:
            return self.extended_objects[sim_obj_name]
        else:
            # Try to find by world lookup
            try:
                return self.world.name_to_body(sim_obj_name)
            except:
                print(f"Object {pddl_obj_name} (sim: {sim_obj_name}) not found in any object dictionary")
                return None

    def _stretch_arm_to_container(self, container):
        """Stretch arm towards container for reaching motion"""
        if container.lower() == 'fridge':
            # Reach towards fridge interior
            reach_pose = [0.5, 0.2, -0.1, -1.2, 0.0, -0.8, 0.0]
        elif container.lower() == 'cabinet':
            # Reach towards cabinet shelf
            reach_pose = [0.8, 0.4, 0.2, -1.0, 0.0, -0.6, 0.0]
        else:
            # Default reaching pose
            reach_pose = SIDE_HOLDING_LEFT_ARM
            
        set_joint_positions(self.robot_body, self.left_joints, reach_pose)
        wait_for_duration(0.8)

    def _attach_object_to_robot(self, pddl_obj_name):
        """Attach object to robot using VLM-TAMP style geometric transforms"""
        # Convert PDDL name to simulator name for object lookup
        sim_obj_name = self._map_pddl_to_sim_name(pddl_obj_name)
        body_id = self._get_object_body_id(pddl_obj_name)
        
        if body_id is None:
            print(f"Cannot attach {pddl_obj_name} (sim: {sim_obj_name}) - object not found")
            return False
            
        try:
            # Step 1: Get current gripper and object poses
            gripper_pose = get_link_pose(self.robot_body, self.gripper_link)
            object_pose = get_pose(body_id)
            print(f"    Gripper pose: {gripper_pose[0][:3]}")
            print(f"    Object pose: {object_pose[0][:3]}")
            
            # Step 2: Calculate relative transform (grasp_pose)
            self.grasp_pose = multiply(invert(gripper_pose), object_pose)
            
            # Step 3: Store attachment info
            self.attached_object_id = body_id
            
            # Step 4: Position object at gripper with small offset
            gripper_position, gripper_orientation = gripper_pose
            object_position = (
                gripper_position[0] + 0.08,  # 8cm forward
                gripper_position[1],
                gripper_position[2]
            )
            
            set_pose(body_id, (object_position, gripper_orientation))
            print(f"    Moved {pddl_obj_name} to gripper position: {object_position[:3]}")
            
            # Step 5: Recalculate grasp pose with new position
            object_pose = get_pose(body_id)
            self.grasp_pose = multiply(invert(gripper_pose), object_pose)
            
            print(f"✅ Attached {pddl_obj_name} to robot gripper with transform")
            return True
            
        except Exception as e:
            print(f"❌ Failed to attach {pddl_obj_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _detach_object_from_robot(self):
        """Detach object from robot gripper"""
        if self.attached_object_id is not None:
            self.attached_object_id = None
            self.grasp_pose = None
            print("✅ Detached object from robot gripper")
            return True
        return False

    def _update_attached_object(self):
        """Update attached object position based on current gripper pose (VLM-TAMP style)"""
        if self.attached_object_id is not None and self.grasp_pose is not None:
            try:
                # Get current gripper pose
                gripper_pose = get_link_pose(self.robot_body, self.gripper_link)
                
                # Calculate new object pose using relative transform
                object_pose = body_from_end_effector(gripper_pose, self.grasp_pose)
                
                # Update object position
                set_pose(self.attached_object_id, object_pose)
                
            except Exception as e:
                print(f"Warning: Failed to update attached object: {e}")
                
    def _stretch_arm_to_region(self, region):
        """Stretch arm towards region for reaching motion"""
        if region.lower() in ['fridge', 'cabinet']:
            # Use container-specific arm poses
            self._stretch_arm_to_container(region)
        elif region.lower() in ['countertop_1', 'countertop_2', 'stove']:
            # Reach towards counter/stove surface
            reach_pose = [0.6, 0.3, 0.0, -1.0, 0.0, -0.7, 0.0]
        elif region.lower() == 'sink':
            # Reach towards sink
            reach_pose = [0.5, 0.2, -0.1, -1.1, 0.0, -0.8, 0.0]
        else:
            # Default reaching pose
            reach_pose = SIDE_HOLDING_LEFT_ARM
            
        if region.lower() not in ['fridge', 'cabinet']:
            set_joint_positions(self.robot_body, self.left_joints, reach_pose)
            wait_for_duration(0.8)

    def _place_object_in_region(self, pddl_obj_name, region):
        """Place object in appropriate region location"""
        sim_obj_name = self._map_pddl_to_sim_name(pddl_obj_name)
        body_id = self._get_object_body_id(pddl_obj_name)
        
        if body_id is None:
            print(f"Cannot place {pddl_obj_name} (sim: {sim_obj_name}) - object not found")
            return False
            
        # Get placement position based on region
        if region.lower() in self.region_object_positions:
            position = self.region_object_positions[region.lower()]
        else:
            print(f"Unknown region: {region}")
            return False
            
        # Set object position
        new_pose = Pose(point=Point(*position))
        set_pose(body_id, new_pose)
        print(f"✅ Placed {pddl_obj_name} in {region} at position {position}")
        return True
    
    def _place_object_in_container(self, pddl_obj_name, container):
        """Place object in appropriate container location"""
        sim_obj_name = self._map_pddl_to_sim_name(pddl_obj_name)
        body_id = self._get_object_body_id(pddl_obj_name)
        
        if body_id is None:
            print(f"Cannot place {pddl_obj_name} (sim: {sim_obj_name}) - object not found")
            return False
            
        # Determine placement position based on container
        if container.lower() == 'fridge':
            # Place in fridge bottom shelf
            position = self.container_positions['fridge']['bottom_shelf']
        elif container.lower() == 'cabinet':
            # Place on cabinet shelf
            position = self.container_positions['cabinet']['shelf']
        else:
            print(f"Unknown container: {container}")
            return False
            
        # Set object position
        new_pose = Pose(point=Point(*position))
        set_pose(body_id, new_pose)
        print(f"✅ Placed {pddl_obj_name} in {container} at position {position}")
        return True

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
        elif action.startswith('putdown_to_container '):
            return self._execute_putdown_to_container(action, verbose)
        elif action.startswith('put_food_in_cookware '):
            return self._execute_put_food_in_cookware(action, verbose)
        elif action.startswith('take_food_from_cookware '):
            return self._execute_take_food_from_cookware(action, verbose)
        elif action.startswith('place_food_on_utensil '):
            return self._execute_place_food_on_utensil(action, verbose)
        elif action.startswith('take_food_from_utensil '):
            return self._execute_take_food_from_utensil(action, verbose)
        elif action.startswith('place_cookware_on_stovetop '):
            return self._execute_place_cookware_on_stovetop(action, verbose)
        elif action.startswith('remove_cookware_from_stovetop '):
            return self._execute_remove_cookware_from_stovetop(action, verbose)
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
            
        # Execute path with attachment updates
        for waypoint in base_path:
            set_base_values(self.robot_body, waypoint)
            # Update attached object position during movement
            self._update_attached_object()
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
        """Execute pickup_from_region action with enhanced arm motion and attachment"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, pddl_obj, region = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"🤖 Picking up {pddl_obj} from {region}")
            
        # Step 1: Stretch arm towards region
        print(f"  Step 1: Reaching towards {region}...")
        self._stretch_arm_to_region(region)
        
        # Step 2: Simulate grasping motion
        print(f"  Step 2: Simulating grasping motion...")
        grasp_pose = [0.3, 0.1, -0.2, -1.4, 0.0, -1.0, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, grasp_pose)
        wait_for_duration(0.5)
        
        # Step 3: Attach object to robot (using PDDL name)
        print(f"  Step 3: Attaching {pddl_obj} to robot gripper...")
        if self._attach_object_to_robot(pddl_obj):
            self.holding_object = pddl_obj  # Store PDDL name
        else:
            return False
            
        # Step 4: Lift arm with object
        print(f"  Step 4: Lifting {pddl_obj}...")
        set_joint_positions(self.robot_body, self.left_joints, TOP_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        print(f"✅ Successfully picked up {pddl_obj} from {region}")
        return True
    
    def _execute_pickup_from_container(self, action: str, verbose: bool) -> bool:
        """Execute pickup_from_container action with enhanced arm motion and manual attachment"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, pddl_obj, container = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"🤖 Picking up {pddl_obj} from {container}")
            
        # Step 1: Stretch arm towards container
        print(f"  Step 1: Stretching arm towards {container}...")
        self._stretch_arm_to_container(container)
        
        # Step 2: Simulate grasping motion
        print(f"  Step 2: Simulating grasping motion...")
        # Lower arm slightly as if reaching into container
        grasp_pose = [0.3, 0.1, -0.2, -1.4, 0.0, -1.0, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, grasp_pose)
        wait_for_duration(0.5)
        
        # Step 3: Manually attach object to robot (using PDDL name)
        print(f"  Step 3: Attaching {pddl_obj} to robot gripper...")
        if self._attach_object_to_robot(pddl_obj):
            self.holding_object = pddl_obj  # Store PDDL name
        else:
            return False
            
        # Step 4: Lift arm with object
        print(f"  Step 4: Lifting {pddl_obj}...")
        set_joint_positions(self.robot_body, self.left_joints, TOP_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        print(f"✅ Successfully picked up {pddl_obj} from {container}")
        return True
    
    def _execute_putdown_to_region(self, action: str, verbose: bool) -> bool:
        """Execute putdown_to_region action with enhanced arm motion and placement"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, pddl_obj, region = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"🤖 Putting down {pddl_obj} to {region}")
            
        # Step 1: Stretch arm towards region
        print(f"  Step 1: Reaching towards {region}...")
        self._stretch_arm_to_region(region)
        
        # Step 2: Lower arm for placement
        print(f"  Step 2: Lowering arm for placement...")
        place_pose = [0.4, 0.2, -0.3, -1.3, 0.0, -0.9, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, place_pose)
        wait_for_duration(0.5)
        
        # Step 3: Detach object from robot
        print(f"  Step 3: Detaching {pddl_obj} from robot...")
        self._detach_object_from_robot()
        
        # Step 4: Place object in region (using PDDL name)
        print(f"  Step 4: Placing {pddl_obj} in {region}...")
        if not self._place_object_in_region(pddl_obj, region):
            return False
            
        # Step 5: Retract arm
        print(f"  Step 5: Retracting arm...")
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        print(f"✅ Successfully put down {pddl_obj} in {region}")
        return True

    def _execute_putdown_to_container(self, action: str, verbose: bool) -> bool:
        """Execute putdown_to_container action with enhanced arm motion and manual placement"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, pddl_obj, container = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"🤖 Putting down {pddl_obj} to {container}")
            
        # Step 1: Stretch arm towards container
        print(f"  Step 1: Stretching arm towards {container}...")
        self._stretch_arm_to_container(container)
        
        # Step 2: Lower arm to placement position
        print(f"  Step 2: Lowering arm for placement...")
        place_pose = [0.4, 0.2, -0.3, -1.3, 0.0, -0.9, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, place_pose)
        wait_for_duration(0.5)
        
        # Step 3: Detach object from robot
        print(f"  Step 3: Detaching {pddl_obj} from robot...")
        self._detach_object_from_robot()
        
        # Step 4: Manually place object in container (using PDDL name)
        print(f"  Step 4: Placing {pddl_obj} in {container}...")
        if not self._place_object_in_container(pddl_obj, container):
            return False
            
        # Step 5: Retract arm
        print(f"  Step 5: Retracting arm...")
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        print(f"✅ Successfully put down {pddl_obj} in {container}")
        return True
    
    def _execute_put_food_in_cookware(self, action: str, verbose: bool) -> bool:
        """Execute put_food_in_cookware action with enhanced manipulation"""
        parts = action.split()
        if len(parts) < 5:
            return False
            
        robot, pddl_food, pddl_cookware, region = parts[1], parts[2], parts[3], parts[4]
        
        if verbose:
            print(f"🍲 Putting {pddl_food} into {pddl_cookware} at {region}")
            
        # Step 1: Stretch arm towards region
        print(f"  Step 1: Reaching towards {region}...")
        self._stretch_arm_to_region(region)
        
        # Step 2: Lower arm over cookware
        print(f"  Step 2: Positioning over {pddl_cookware}...")
        place_pose = [0.4, 0.2, -0.2, -1.2, 0.0, -0.8, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, place_pose)
        wait_for_duration(0.5)
        
        # Step 3: Detach food from robot
        print(f"  Step 3: Releasing {pddl_food}...")
        self._detach_object_from_robot()
        
        # Step 4: Place food inside cookware (using PDDL names)
        print(f"  Step 4: Placing {pddl_food} in {pddl_cookware}...")
        if not self._place_food_in_cookware(pddl_food, pddl_cookware, region):
            return False
            
        # Step 5: Retract arm
        print(f"  Step 5: Retracting arm...")
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        print(f"✅ Successfully put {pddl_food} into {pddl_cookware}")
        return True
        
    def _execute_take_food_from_cookware(self, action: str, verbose: bool) -> bool:
        """Execute take_food_from_cookware action with enhanced manipulation"""
        parts = action.split()
        if len(parts) < 5:
            return False
            
        robot, pddl_food, pddl_cookware, region = parts[1], parts[2], parts[3], parts[4]
        
        if verbose:
            print(f"🍲 Taking {pddl_food} from {pddl_cookware} at {region}")
            
        # Step 1: Stretch arm towards region
        print(f"  Step 1: Reaching towards {region}...")
        self._stretch_arm_to_region(region)
        
        # Step 2: Lower arm into cookware
        print(f"  Step 2: Reaching into {pddl_cookware}...")
        grasp_pose = [0.3, 0.1, -0.3, -1.4, 0.0, -1.0, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, grasp_pose)
        wait_for_duration(0.5)
        
        # Step 3: Attach food to robot (using PDDL name)
        print(f"  Step 3: Grasping {pddl_food}...")
        if self._attach_object_to_robot(pddl_food):
            self.holding_object = pddl_food
        else:
            return False
            
        # Step 4: Lift food from cookware
        print(f"  Step 4: Lifting {pddl_food} from {pddl_cookware}...")
        set_joint_positions(self.robot_body, self.left_joints, TOP_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        print(f"✅ Successfully took {pddl_food} from {pddl_cookware}")
        return True
        
    def _execute_place_food_on_utensil(self, action: str, verbose: bool) -> bool:
        """Execute place_food_on_utensil action with enhanced manipulation"""
        parts = action.split()
        if len(parts) < 5:
            return False
            
        robot, pddl_food, pddl_utensil, region = parts[1], parts[2], parts[3], parts[4]
        
        if verbose:
            print(f"🍽️ Placing {pddl_food} on {pddl_utensil} at {region}")
            
        # Step 1: Stretch arm towards region
        print(f"  Step 1: Reaching towards {region}...")
        self._stretch_arm_to_region(region)
        
        # Step 2: Lower arm over utensil
        print(f"  Step 2: Positioning over {pddl_utensil}...")
        place_pose = [0.4, 0.2, -0.1, -1.1, 0.0, -0.7, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, place_pose)
        wait_for_duration(0.5)
        
        # Step 3: Detach food from robot
        print(f"  Step 3: Releasing {pddl_food}...")
        self._detach_object_from_robot()
        
        # Step 4: Place food on utensil (using PDDL names)
        print(f"  Step 4: Placing {pddl_food} on {pddl_utensil}...")
        if not self._place_food_on_utensil_position(pddl_food, pddl_utensil, region):
            return False
            
        # Step 5: Retract arm
        print(f"  Step 5: Retracting arm...")
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        print(f"✅ Successfully placed {pddl_food} on {pddl_utensil}")
        return True
        
    def _execute_take_food_from_utensil(self, action: str, verbose: bool) -> bool:
        """Execute take_food_from_utensil action with enhanced manipulation"""
        parts = action.split()
        if len(parts) < 5:
            return False
            
        robot, pddl_food, pddl_utensil, region = parts[1], parts[2], parts[3], parts[4]
        
        if verbose:
            print(f"🍽️ Taking {pddl_food} from {pddl_utensil} at {region}")
            
        # Step 1: Stretch arm towards region
        print(f"  Step 1: Reaching towards {region}...")
        self._stretch_arm_to_region(region)
        
        # Step 2: Lower arm to utensil level
        print(f"  Step 2: Reaching to {pddl_utensil}...")
        grasp_pose = [0.3, 0.1, -0.2, -1.3, 0.0, -0.9, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, grasp_pose)
        wait_for_duration(0.5)
        
        # Step 3: Attach food to robot (using PDDL name)
        print(f"  Step 3: Grasping {pddl_food}...")
        if self._attach_object_to_robot(pddl_food):
            self.holding_object = pddl_food
        else:
            return False
            
        # Step 4: Lift food from utensil
        print(f"  Step 4: Lifting {pddl_food} from {pddl_utensil}...")
        set_joint_positions(self.robot_body, self.left_joints, TOP_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        print(f"✅ Successfully took {pddl_food} from {pddl_utensil}")
        return True
        
    def _execute_place_cookware_on_stovetop(self, action: str, verbose: bool) -> bool:
        """Execute place_cookware_on_stovetop action with enhanced manipulation"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, pddl_cookware, stovetop = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"🔥 Placing {pddl_cookware} on {stovetop}")
            
        # Step 1: Stretch arm towards stovetop
        print(f"  Step 1: Reaching towards {stovetop}...")
        self._stretch_arm_to_region('stove')
        
        # Step 2: Lower arm to stovetop level
        print(f"  Step 2: Lowering {pddl_cookware} to stovetop...")
        place_pose = [0.4, 0.3, -0.2, -1.1, 0.0, -0.7, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, place_pose)
        wait_for_duration(0.5)
        
        # Step 3: Detach cookware from robot
        print(f"  Step 3: Releasing {pddl_cookware}...")
        self._detach_object_from_robot()
        
        # Step 4: Place cookware on stovetop (using PDDL name)
        print(f"  Step 4: Positioning {pddl_cookware} on stovetop...")
        if not self._place_cookware_on_stovetop_position(pddl_cookware):
            return False
            
        # Step 5: Retract arm
        print(f"  Step 5: Retracting arm...")
        set_joint_positions(self.robot_body, self.left_joints, SIDE_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        self.holding_object = None
        print(f"✅ Successfully placed {pddl_cookware} on stovetop")
        return True
        
    def _execute_remove_cookware_from_stovetop(self, action: str, verbose: bool) -> bool:
        """Execute remove_cookware_from_stovetop action with enhanced manipulation"""
        parts = action.split()
        if len(parts) < 4:
            return False
            
        robot, pddl_cookware, stovetop = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"🔥 Removing {pddl_cookware} from {stovetop}")
            
        # Step 1: Stretch arm towards stovetop
        print(f"  Step 1: Reaching towards {stovetop}...")
        self._stretch_arm_to_region('stove')
        
        # Step 2: Lower arm to cookware level
        print(f"  Step 2: Grasping {pddl_cookware}...")
        grasp_pose = [0.3, 0.2, -0.2, -1.2, 0.0, -0.8, 0.0]
        set_joint_positions(self.robot_body, self.left_joints, grasp_pose)
        wait_for_duration(0.5)
        
        # Step 3: Attach cookware to robot (using PDDL name)
        print(f"  Step 3: Lifting {pddl_cookware}...")
        if self._attach_object_to_robot(pddl_cookware):
            self.holding_object = pddl_cookware
        else:
            return False
            
        # Step 4: Lift cookware from stovetop
        print(f"  Step 4: Removing {pddl_cookware} from stovetop...")
        set_joint_positions(self.robot_body, self.left_joints, TOP_HOLDING_LEFT_ARM)
        wait_for_duration(0.5)
        
        print(f"✅ Successfully removed {pddl_cookware} from stovetop")
        return True
        
    def _place_food_in_cookware(self, pddl_food_name, pddl_cookware_name, region):
        """Place food inside cookware at specified region (using PDDL names)"""
        food_id = self._get_object_body_id(pddl_food_name)
        cookware_id = self._get_object_body_id(pddl_cookware_name)
        
        if food_id is None or cookware_id is None:
            print(f"Cannot place {pddl_food_name} in {pddl_cookware_name} - objects not found")
            return False
            
        # Get cookware position and add offset for food inside
        cookware_pose = get_pose(cookware_id)
        cookware_position = cookware_pose[0]
        
        # Place food inside cookware (slightly above bottom)
        food_position = (
            cookware_position[0] + self.cookware_food_offset[0],
            cookware_position[1] + self.cookware_food_offset[1],
            cookware_position[2] + self.cookware_food_offset[2]
        )
        
        set_pose(food_id, (food_position, cookware_pose[1]))
        print(f"✅ Placed {pddl_food_name} inside {pddl_cookware_name} at {food_position[:3]}")
        return True
        
    def _place_food_on_utensil_position(self, pddl_food_name, pddl_utensil_name, region):
        """Place food on utensil at specified region (using PDDL names)"""
        food_id = self._get_object_body_id(pddl_food_name)
        utensil_id = self._get_object_body_id(pddl_utensil_name)
        
        if food_id is None or utensil_id is None:
            print(f"Cannot place {pddl_food_name} on {pddl_utensil_name} - objects not found")
            return False
            
        # Get utensil position and add offset for food on top
        utensil_pose = get_pose(utensil_id)
        utensil_position = utensil_pose[0]
        
        # Place food on utensil (slightly above surface)
        food_position = (
            utensil_position[0] + self.utensil_food_offset[0],
            utensil_position[1] + self.utensil_food_offset[1],
            utensil_position[2] + self.utensil_food_offset[2]
        )
        
        set_pose(food_id, (food_position, utensil_pose[1]))
        print(f"✅ Placed {pddl_food_name} on {pddl_utensil_name} at {food_position[:3]}")
        return True
        
    def _place_cookware_on_stovetop_position(self, pddl_cookware_name):
        """Place cookware on stovetop at fixed position (using PDDL name)"""
        cookware_id = self._get_object_body_id(pddl_cookware_name)
        if cookware_id is None:
            print(f"Cannot place {pddl_cookware_name} - cookware not found")
            return False
            
        # Set cookware position on stovetop
        new_pose = Pose(point=Point(*self.stovetop_cookware_position))
        set_pose(cookware_id, new_pose)
        print(f"✅ Placed {pddl_cookware_name} on stovetop at {self.stovetop_cookware_position}")
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
            
        stove, pddl_cookware, pddl_food = parts[1], parts[2], parts[3]
        
        if verbose:
            print(f"Waiting for {pddl_food} to cook in {pddl_cookware} on {stove}")
            
        # Simulate cooking time
        wait_for_duration(2.0)
        
        return True
    
    def __del__(self):
        """Cleanup PyBullet connection when object is destroyed."""
        try:
            # Clean up any attachments
            self._detach_object_from_robot()
            disconnect()
        except:
            pass


def main():
    """Main function that creates executor and runs test actions."""

    # test_actions = [
    #     "move pr2 kitchen_corner fridge",
    #     "pickup_from_container pr2 chicken fridge",
    #     "move pr2 fridge countertop_2",
    #     "put_food_in_cookware pr2 chicken pot countertop_2",
    #     "pickup_from_region pr2 pot countertop_2",
    #     "move pr2 countertop_2 stove",
    #     "place_cookware_on_stovetop pr2 pot stove",
    # ]

    test_actions = [
        "move pr2 kitchen_corner fridge",
        "pickup_from_container pr2 chicken fridge",
    ]
    
    try:
        print("=== TESTING SIMULATOR EXECUTOR V5 ===")
        print("Testing PDDL name mapping functionality...")
        
        # Create executor (this initializes everything)
        executor = SimulatorExecutor()
        
        print(f"\nTesting with {len(test_actions)} actions:")
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