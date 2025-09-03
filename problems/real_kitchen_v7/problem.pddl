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
    (at pot countertop)
    (at pan countertop)

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
  
  ; (:goal
  ;   (and
  ;     (on_utensil pancake plate)
  ;     (on_utensil apple plate)
  ;     (cooked pancake)
  ;     (at pan sink)
  ;     (at pot countertop)
  ;     (at plate countertop)
  ;     (door_closed fridge)
  ;     (door_closed cabinet)
  ;     (turned_off stove)
  ;   )
  ; )


  (:goal
    (and
      (on_utensil chicken plate)
      (on_utensil potato plate)
      (cooked chicken)
      (cooked potato)
      (at bowl sink)
      (at pot sink)
      (at pan countertop)
      (door_closed fridge)
      (door_closed cabinet)
      (turned_off stove)
    )
  )

  ; (:goal
  ;   (and
  ;     (on_utensil fish plate)
  ;     (on_utensil bread plate)
  ;     (on_utensil salt plate)
  ;     (on_utensil butter plate)
  ;     (cooked fish)
  ;     (at bowl sink)
  ;     (at pan sink)
  ;     (at pot countertop)
  ;     (door_closed fridge)
  ;     (door_closed cabinet)
  ;     (turned_off stove)
  ;   )
  ; )


)