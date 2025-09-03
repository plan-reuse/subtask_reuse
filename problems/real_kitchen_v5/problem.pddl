(define (problem kitchen-problem)
  (:domain kitchen)
  (:objects
    stretch_robot - robot
    kitchen_corner countertop sink - region
    fridge cabinet - container
    stove - stovetop
    bowl plate - utensil
    pot pan - cookware
    chicken potato cabbage salt - food
  )
  (:init
    (at_robot stretch_robot kitchen_corner)
    (at bowl countertop)
    (at plate sink)
    (at pot countertop)
    (at pan sink)
    (inside chicken fridge)
    (inside potato fridge)
    (inside cabbage fridge)
    (inside salt cabinet)
    (door_closed fridge)
    (door_closed cabinet)
    (turned_off stove)
    (handempty stretch_robot)
  )
  (:goal
    (and
      (at bowl sink)
      (at plate countertop)
      (at pot countertop)
      (at pan sink)
      (on_utensil chicken plate)
      (on_utensil potato plate)
      (cooked chicken)
      (cooked potato)
      (door_closed fridge)
      (door_closed cabinet)
      (turned_off stove)
    )
  )
)