(define (domain kitchen)
  (:requirements :strips :typing)
  
  (:types robot region utensil)

  (:predicates
    (at_robot ?r - robot ?l - region)
    (at_obj ?o - utensil ?l - region)
    (holding ?r - robot ?o - utensil)
    (handempty ?r - robot)
  )

  (:action move
    :parameters (?r - robot ?from - region ?to - region)
    :precondition (and (at_robot ?r ?from))
    :effect (and (not (at_robot ?r ?from)) (at_robot ?r ?to))
  )

  (:action pickup
    :parameters (?r - robot ?o - utensil ?l - region)
    :precondition (and (at_robot ?r ?l) (at_obj ?o ?l) (handempty ?r))
    :effect (and (holding ?r ?o)
                 (not (at_obj ?o ?l))
                 (not (handempty ?r)))
  )

  (:action putdown
    :parameters (?r - robot ?o - utensil ?l - region)
    :precondition (and (at_robot ?r ?l) (holding ?r ?o))
    :effect (and (at_obj ?o ?l)
                 (handempty ?r)
                 (not (holding ?r ?o)))
  )
)
