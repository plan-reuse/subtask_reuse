

(define (domain kitchen)
  (:requirements :strips :typing)
  
  (:types robot location object)

  (:predicates
    (at ?r - robot ?l - location)
    (at_obj ?o - object ?l - location)
    (holding ?r - robot ?o - object)
    (handempty ?r - robot)
  )

  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and (at ?r ?from))
    :effect (and (not (at ?r ?from)) (at ?r ?to))
  )

  (:action pickup
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and (at ?r ?l) (at_obj ?o ?l) (handempty ?r))
    :effect (and (holding ?r ?o)
                 (not (at_obj ?o ?l))
                 (not (handempty ?r)))
  )

  (:action putdown
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and (at ?r ?l) (holding ?r ?o))
    :effect (and (at_obj ?o ?l)
                 (handempty ?r)
                 (not (holding ?r ?o)))
  )
)
