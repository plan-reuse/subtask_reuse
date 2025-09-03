# knowledge_base_script.py

# Define the knowledge base
knowledge_base = {
    (
        "goal: (and (at <cookware> countertop) (door_closed <container>))",
        "robot_init: (at_robot stretch_robot kitchen_corner)"
    ): [
        "move stretch_robot kitchen_corner <container>",
        "open_container stretch_robot <container>",
        "pickup_from_container stretch_robot <cookware> <container>",
        "move stretch_robot <container> countertop",
        "putdown_to_region stretch_robot <cookware> countertop",
        "move stretch_robot countertop <container>",
        "close_container stretch_robot <container>"
    ],

    (
        "goal: (and (in_cookware <food> <cookware>) (door_closed <container>))",
        "robot_init: (at_robot stretch_robot cabinet)"
    ): [
        "move stretch_robot cabinet <container>",
        "open_container stretch_robot <container>",
        "pickup_from_container stretch_robot <food> <container>",
        "move stretch_robot <container> countertop",
        "put_food_in_cookware stretch_robot <food> <cookware> countertop",
        "move stretch_robot countertop <container>",
        "close_container stretch_robot <container>"
    ]
}

# Function to retrieve actions
def get_actions(goal, robot_init):
    key = (f"goal: {goal}", f"robot_init: {robot_init}")
    return knowledge_base.get(key, None)

# Main logic with sample input
if __name__ == "__main__":
    # Sample 1
    goal = "(and (at <cookware> countertop) (door_closed <container>))"
    robot_init = "(at_robot stretch_robot kitchen_corner)"
    
    # Sample 2
    # goal = "(and (in_cookware <food> <cookware>) (door_closed <container>))"
    # robot_init = "(at_robot stretch_robot cabinet)"

    actions = get_actions(goal, robot_init)

    print(f"\nGoal: {goal}")
    print(f"Robot Init: {robot_init}")

    if actions:
        print("\nRetrieved Action Plan:")
        for i, action in enumerate(actions, 1):
            print(f"{i}. {action}")
    else:
        print("\nNo plan found for the given goal and robot_init.")
