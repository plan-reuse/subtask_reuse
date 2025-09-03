(define (problem kitchen-problem)
  (:domain kitchen)
  (:objects
    stretch_robot - robot
    kitchen_corner countertop sink - region
    fridge cabinet - container
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
    (handempty stretch_robot)
  )
  (:goal
    (and
      (at cup sink)
      (at plate countertop)
      (at chicken countertop)
      (at potato countertop)
      (door_closed fridge)
      (door_closed cabinet)
    )
  )
)


; move stretch_robot kitchen_corner fridge (1)
; open_container stretch_robot fridge (1)
; pickup_from_container stretch_robot chicken fridge (1)
; move stretch_robot fridge countertop (1)
; putdown_to_region stretch_robot chicken countertop (1)
; move stretch_robot countertop fridge (1)
; pickup_from_container stretch_robot potato fridge (1)
; move stretch_robot fridge countertop (1)
; putdown_to_region stretch_robot potato countertop (1)
; pickup_from_region stretch_robot cup countertop (1)
; move stretch_robot countertop sink (1)
; putdown_to_region stretch_robot cup sink (1)
; pickup_from_region stretch_robot plate sink (1)
; move stretch_robot sink countertop (1)
; putdown_to_region stretch_robot plate countertop (1)
; move stretch_robot countertop fridge (1)
; close_container stretch_robot fridge (1)