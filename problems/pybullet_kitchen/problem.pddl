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
    (at_robot pr2 kitchen_corner)
    (handempty pr2)

    (at plate countertop_1)
    (at pot countertop_2)

    (inside salt cabinet)
    (inside pepper cabinet)
    (inside oil cabinet)
    (inside vinegar cabinet)
    (door_closed cabinet)

    (inside chicken fridge)
    (inside potato fridge)
    (inside tomato fridge)
    (inside cabbage fridge)
    (inside zucchini fridge)
    (inside artichoke fridge)
    (inside milk fridge)
    (door_closed fridge)

    (turned_off stove)
  )
  
  (:goal
    (and
      (cooked chicken)

    )
  )

)

; 