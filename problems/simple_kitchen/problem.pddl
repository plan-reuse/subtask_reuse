

(define (problem kitchen-problem)
  (:domain kitchen)

  (:objects
    robot1 - robot
    kitchen table sink - location
    cup plate - object
  )

  (:init
    (at robot1 kitchen)
    (at_obj cup table)
    (at_obj plate sink)
    (handempty robot1)
  )

  (:goal
    (and
      (at_obj cup sink)
      (at_obj plate table)
    )
  )
)
