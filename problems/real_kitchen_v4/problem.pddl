(define (problem kitchen-problem)
  (:domain kitchen)
  (:objects
    stretch_robot - robot
    kitchen_corner countertop sink - region
    fridge cabinet - container
    oven - appliance
    cup plate - utensil
    chicken potato cabbage salt - food
  )
  (:init
    (at_robot stretch_robot kitchen_corner)
    (at cup countertop)
    (at plate sink)
    (inside chicken fridge)
    (inside potato fridge)
    (inside cabbage fridge)
    (inside salt cabinet)
    (door_closed fridge)
    (door_closed cabinet)
    (door_closed oven)
    (turned_off oven)
    (handempty stretch_robot)
  )
  (:goal
    (and
      (at cup sink)
      (at plate countertop)
      (at chicken countertop)
      (at potato countertop)
      (cooked chicken)
      (cooked potato)
      (door_closed fridge)
      (door_closed cabinet)
      (door_closed oven)
      (turned_off oven)
    )
  )
)
