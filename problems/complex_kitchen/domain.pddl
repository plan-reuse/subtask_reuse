(define (domain kitchen)
  (:requirements :strips :typing)

  (:types robot location object utensil appliance ingredient)

  (:predicates
    (at ?r - robot ?l - location)
    (at_obj ?o - object ?l - location)
    (holding ?r - robot ?o - object)
    (handempty ?r - robot)
    (clean ?o - object)
    (dirty ?o - object)
    (chopped ?i - ingredient)
    (cooked ?i - ingredient)
    (turned_on ?a - appliance)
    (open ?a - appliance)
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

  (:action wash
    :parameters (?r - robot ?u - utensil ?a - appliance ?l - location)
    :precondition (and (at ?r ?l) (holding ?r ?u) (at_obj ?a ?l) (turned_on ?a) (dirty ?u))
    :effect (and (clean ?u) (not (dirty ?u)))
  )

  (:action dry
    :parameters (?r - robot ?u - utensil ?l - location)
    :precondition (and (at ?r ?l) (holding ?r ?u) (clean ?u))
    :effect (and (not (clean ?u)))
  )

  (:action turn_on
    :parameters (?r - robot ?a - appliance ?l - location)
    :precondition (and (at ?r ?l) (at_obj ?a ?l))
    :effect (turned_on ?a)
  )

  (:action open_appliance
    :parameters (?r - robot ?a - appliance ?l - location)
    :precondition (and (at ?r ?l) (at_obj ?a ?l))
    :effect (open ?a)
  )

  (:action close_appliance
    :parameters (?r - robot ?a - appliance ?l - location)
    :precondition (and (at ?r ?l) (at_obj ?a ?l) (open ?a))
    :effect (not (open ?a))
  )

  (:action chop
    :parameters (?r - robot ?i - ingredient ?u - utensil ?l - location)
    :precondition (and (at ?r ?l) (holding ?r ?u) (at_obj ?i ?l))
    :effect (chopped ?i)
  )

  (:action cook
    :parameters (?r - robot ?i - ingredient ?a - appliance ?l - location)
    :precondition (and (at ?r ?l) (at_obj ?i ?l) (turned_on ?a) (open ?a) (chopped ?i))
    :effect (cooked ?i)
  )
)
