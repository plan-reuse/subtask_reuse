(define (problem breakfast-preparation)
  (:domain kitchen)

  (:objects
    pr2 - robot

    kitchen_corner countertop_1 countertop_2 sink - surface
    fridge cabinet - container
    stove_1 - stovetop

    plate - utensil
    pot   - cookware

    salt pepper - condiment
    oil vinegar milk - liquid
    chicken potato tomato cabbage zucchini artichoke - grocery
  )

  (:init
    (at_robot pr2 kitchen_corner)
    (handempty pr2)

    (at plate countertop_1)
    (at pot   countertop_2)

    (inside salt cabinet)
    (inside pepper cabinet)
    (inside oil cabinet)
    (inside vinegar cabinet)

    (inside chicken fridge)
    (inside potato  fridge)
    (inside tomato  fridge)
    (inside cabbage fridge)
    (inside zucchini fridge)
    (inside artichoke fridge)
    (inside milk fridge)

    (door_closed cabinet)
    (door_closed fridge)

    (turned_off stove_1)
  )

  (:goal
    (and
    (poured oil pot)
    (inside oil cabinet)
    ;(sprinkled salt pot)
    (cooked chicken)
    ;(cooked potato)
    (on_utensil chicken plate)
    ;(on_utensil potato plate)
    )
  )
)


;low
;(cooked potato)

;mid
;(poured oil pot)
;(inside oil cabinet)
;(cooked potato)

;(poured oil pot)
;(cooked potato)

;high
;(poured oil pot)
;(inside oil cabinet)
;(cooked chicken)
;(on_utensil chicken plate)

;(poured oil pot)
;(cooked chicken)
;(on_utensil chicken plate)


    ;(poured oil pot)
    ;(sprinkled salt pot)
    ;(cooked chicken)
    ;(cooked potato)
    ;(on_utensil chicken plate)
    ;(on_utensil potato plate)