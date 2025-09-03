(define (problem breakfast-preparation)
  (:domain kitchen)
  
  (:objects
    stretch_robot - robot
    kitchen_corner countertop sink - region
    fridge cabinet - container
    stove - stovetop
    bowl plate - utensil
    pot pan - cookware
    pancake bread salt butter chicken fish potato tomato cabbage lettuce apple - food
  )

  (:init
    (at_robot stretch_robot kitchen_corner)
    (handempty stretch_robot)

    (at bowl countertop)
    (at plate countertop)

    (inside pot cabinet)
    (inside pan cabinet)

    (inside pancake cabinet)
    (inside bread cabinet)
    (inside salt cabinet)
    (inside butter cabinet)

    (inside chicken fridge)
    (inside fish fridge)
    (inside potato fridge)
    (inside tomato fridge)
    (inside cabbage fridge)
    (inside lettuce fridge)
    (inside apple fridge)
    (door_closed fridge)

    (door_closed cabinet)
    (turned_off stove)
  )
  
  (:goal

  

  )

)