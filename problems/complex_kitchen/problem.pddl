(define (problem kitchen-big-problem)
  (:domain kitchen)

(:objects
  robot1 - robot
  kitchen table sink counter stove fridge pantry shelf - location
  cup plate bowl spoon fork knife pan pot - utensil
  dishwasher oven microwave - appliance
  carrot potato tomato chicken - ingredient
)


  (:init
    (at robot1 kitchen)
    ;; utensils
    (at_obj cup table)
    (at_obj plate sink)
    (at_obj bowl counter)
    (at_obj spoon counter)
    (at_obj fork counter)
    (at_obj knife counter)
    (at_obj pan shelf)
    (at_obj pot shelf)

    ;; appliances
    (at_obj dishwasher sink)
    (at_obj oven counter)
    (at_obj microwave counter)
    (at_obj fridge kitchen)
    (at_obj stove kitchen)

    ;; ingredients
    (at_obj carrot fridge)
    (at_obj potato fridge)
    (at_obj tomato fridge)
    (at_obj chicken fridge)

    ;; states
    (dirty cup)
    (dirty plate)
    (dirty bowl)
    (dirty spoon)
    (dirty fork)
    (dirty knife)
    (handempty robot1)
  )

  (:goal
    (and
      (cooked carrot)
      (cooked potato)
      (cooked tomato)
      (cooked chicken)
    )
  )
)

; (cooked carrot)
; move robot1 kitchen counter (1)
; pickup robot1 bowl counter (1)
; open_appliance robot1 microwave counter (1)
; turn_on robot1 microwave counter (1)
; move robot1 counter fridge (1)
; chop robot1 carrot bowl fridge (1)
; cook robot1 carrot microwave fridge (1)

; (cooked potato)
; move robot1 kitchen counter (1)
; pickup robot1 bowl counter (1)
; open_appliance robot1 microwave counter (1)
; turn_on robot1 microwave counter (1)
; move robot1 counter fridge (1)
; chop robot1 potato bowl fridge (1)
; cook robot1 potato microwave fridge (1)


; (cooked carrot)
; (cooked potato)

; move robot1 kitchen counter (1)
; pickup robot1 bowl counter (1)
; open_appliance robot1 microwave counter (1)
; turn_on robot1 microwave counter (1)
; move robot1 counter fridge (1)
; chop robot1 carrot bowl fridge (1)
; cook robot1 carrot microwave fridge (1)

; chop robot1 potato bowl fridge (1)
; cook robot1 potato microwave fridge (1)



; (cooked carrot)
; (cooked potato)
; (cooked tomato)
; (cooked chicken)

; move robot1 kitchen counter (1)
; pickup robot1 bowl counter (1)
; open_appliance robot1 microwave counter (1)
; turn_on robot1 microwave counter (1)
; move robot1 counter fridge (1)
; chop robot1 carrot bowl fridge (1)
; cook robot1 carrot microwave fridge (1)

; chop robot1 chicken bowl fridge (1)
; cook robot1 chicken microwave fridge (1)

; chop robot1 potato bowl fridge (1)
; cook robot1 potato microwave fridge (1)

; chop robot1 tomato bowl fridge (1)
; cook robot1 tomato microwave fridge (1)