(define (problem kitchen-problem)
  (:domain kitchen)

  (:objects
    stretch_robot - robot
    kitchen_corner countertop sink fridge cabinet - region
    cup plate - utensil
  )

  (:init
    (at_robot stretch_robot kitchen_corner)
    (at_obj cup countertop)
    (at_obj plate sink)
    (handempty stretch_robot)
  )

  (:goal
    (and
      (at_obj cup sink)
      (at_obj plate countertop)
    )
  )
)
