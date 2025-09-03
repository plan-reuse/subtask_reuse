(define (problem breakfast-preparation)
  (:domain kitchen)
  (:objects
    pr2 - robot
    kitchen_corner countertop_1 countertop_2 sink - region
    fridge cabinet - container
    stove - stovetop
    plate - utensil
    pot - cookware
    salt pepper chicken potato tomato cabbage zucchini artichoke oil vinegar milk - food
  )
  (:init
    (at plate countertop_1)
    (at pot countertop_2)
    (at_robot pr2 countertop_2)
    (door_open cabinet)
    (door_open fridge)
    (handempty pr2)
    (in_cookware chicken pot)
    (inside artichoke fridge)
    (inside cabbage fridge)
    (inside milk fridge)
    (inside oil cabinet)
    (inside pepper cabinet)
    (inside potato fridge)
    (inside salt cabinet)
    (inside tomato fridge)
    (inside vinegar cabinet)
    (inside zucchini fridge)
    (turned_off stove)
  )
  
  (:goal
    (and (at_robot pr2 cabinet) (holding pr2 salt))
  )

)

