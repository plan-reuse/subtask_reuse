"""Action effects parsing utilities."""

from typing import List


def parse_action_predicates(action: str) -> List[str]:
    """Parse a PDDL action to extract the predicates it affects based on the domain."""
    effects = []
    
    # Clean the action string
    action = action.strip()
    
    # Movement effects (domain uses at_robot predicate)
    if action.startswith("move "):
        parts = action.split()
        if len(parts) >= 4:
            robot, from_loc, to_loc = parts[1], parts[2], parts[3]
            # Normalize robot name for consistency
            normalized_robot = "pr2"
            effects.append(f"(not (at_robot {normalized_robot} {from_loc}))")
            effects.append(f"(at_robot {normalized_robot} {to_loc})")
    
    # Container opening effects
    elif action.startswith("open_container "):
        parts = action.split()
        if len(parts) >= 3:
            robot, container = parts[1], parts[2]
            effects.append(f"(door_open {container})")
            effects.append(f"(not (door_closed {container}))")
    
    # Container closing effects
    elif action.startswith("close_container "):
        parts = action.split()
        if len(parts) >= 3:
            robot, container = parts[1], parts[2]
            effects.append(f"(door_closed {container})")
            effects.append(f"(not (door_open {container}))")
    
    # Pickup from region
    elif action.startswith("pickup_from_region "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, location = parts[1], parts[2], parts[3]
            normalized_robot = "pr2"
            effects.append(f"(holding {normalized_robot} {obj})")
            effects.append(f"(not (at {obj} {location}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Putdown to region
    elif action.startswith("putdown_to_region "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, location = parts[1], parts[2], parts[3]
            normalized_robot = "pr2"
            effects.append(f"(at {obj} {location})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {obj}))")
    
    # Pickup from container
    elif action.startswith("pickup_from_container "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, container = parts[1], parts[2], parts[3]
            normalized_robot = "pr2"
            effects.append(f"(holding {normalized_robot} {obj})")
            effects.append(f"(not (inside {obj} {container}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Putdown to container
    elif action.startswith("putdown_to_container "):
        parts = action.split()
        if len(parts) >= 4:
            robot, obj, container = parts[1], parts[2], parts[3]
            normalized_robot = "pr2"
            effects.append(f"(inside {obj} {container})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {obj}))")
    
    # Put food in cookware
    elif action.startswith("put_food_in_cookware "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "pr2"
            effects.append(f"(in_cookware {food} {cookware})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {food}))")
    
    # Take food from cookware
    elif action.startswith("take_food_from_cookware "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, cookware, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "pr2"
            effects.append(f"(holding {normalized_robot} {food})")
            effects.append(f"(not (in_cookware {food} {cookware}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Place food on utensil
    elif action.startswith("place_food_on_utensil "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, utensil, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "pr2"
            effects.append(f"(on_utensil {food} {utensil})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {food}))")
    
    # Take food from utensil
    elif action.startswith("take_food_from_utensil "):
        parts = action.split()
        if len(parts) >= 5:
            robot, food, utensil, location = parts[1], parts[2], parts[3], parts[4]
            normalized_robot = "pr2"
            effects.append(f"(holding {normalized_robot} {food})")
            effects.append(f"(not (on_utensil {food} {utensil}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Place cookware on stovetop
    elif action.startswith("place_cookware_on_stovetop "):
        parts = action.split()
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            normalized_robot = "pr2"
            effects.append(f"(on_stovetop {cookware} {stovetop})")
            effects.append(f"(handempty {normalized_robot})")
            effects.append(f"(not (holding {normalized_robot} {cookware}))")
    
    # Remove cookware from stovetop
    elif action.startswith("remove_cookware_from_stovetop "):
        parts = action.split()
        if len(parts) >= 4:
            robot, cookware, stovetop = parts[1], parts[2], parts[3]
            normalized_robot = "pr2"
            effects.append(f"(holding {normalized_robot} {cookware})")
            effects.append(f"(not (on_stovetop {cookware} {stovetop}))")
            effects.append(f"(not (handempty {normalized_robot}))")
    
    # Turn on stove
    elif action.startswith("turn_on_stove "):
        parts = action.split()
        if len(parts) >= 3:
            robot, stove = parts[1], parts[2]
            effects.append(f"(turned_on {stove})")
            effects.append(f"(not (turned_off {stove}))")
    
    # Turn off stove
    elif action.startswith("turn_off_stove "):
        parts = action.split()
        if len(parts) >= 3:
            robot, stove = parts[1], parts[2]
            effects.append(f"(turned_off {stove})")
            effects.append(f"(not (turned_on {stove}))")
    
    # Wait for food to cook
    elif action.startswith("wait_for_food_to_cook "):
        parts = action.split()
        if len(parts) >= 4:
            stovetop, cookware, food = parts[1], parts[2], parts[3]
            effects.append(f"(cooked {food})")
    
    return effects