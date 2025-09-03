(define (problem kitchen-problem)
  (:domain kitchen)
  (:objects
    stretch_robot - robot
    kitchen_corner countertop sink fridge cabinet - region
    cup plate - utensil
    chicken potato cabbage salt - food
  )

  (:init
    (at_robot stretch_robot kitchen_corner)
    (at_utensil cup countertop)
    (at_utensil plate sink)
    (at_food chicken fridge)
    (at_food potato fridge)
    (at_food cabbage fridge)
    (at_food salt cabinet)
    (handempty stretch_robot)
  )

  (:goal
    (and
      (at_utensil cup sink)
      (at_utensil plate countertop)
      (at_food chicken countertop)
      (at_food potato countertop)
    )
  )
)

; move stretch_robot kitchen_corner fridge (1)
; pickup_food stretch_robot chicken fridge (1)
; move stretch_robot fridge countertop (1)
; putdown_food stretch_robot chicken countertop (1)

; move stretch_robot countertop fridge (1)
; pickup_food stretch_robot potato fridge (1)
; move stretch_robot fridge countertop (1)
; putdown_food stretch_robot potato countertop (1)

; pickup_utensil stretch_robot cup countertop (1)
; move stretch_robot countertop sink (1)
; putdown_utensil stretch_robot cup sink (1)

; pickup_utensil stretch_robot plate sink (1)
; move stretch_robot sink countertop (1)
; putdown_utensil stretch_robot plate countertop (1)