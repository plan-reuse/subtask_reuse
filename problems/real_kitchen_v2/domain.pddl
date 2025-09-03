
(define (domain kitchen)
  (:requirements :strips :typing)
  
  (:types robot region utensil food)
  (:predicates
    (at_robot ?r - robot ?l - region)
    (at_utensil ?u - utensil ?l - region)
    (at_food ?f - food ?l - region)
    (holding_utensil ?r - robot ?u - utensil)
    (holding_food ?r - robot ?f - food)
    (handempty ?r - robot)
  )
  (:action move
    :parameters (?r - robot ?from - region ?to - region)
    :precondition (and (at_robot ?r ?from))
    :effect (and (not (at_robot ?r ?from)) (at_robot ?r ?to))
  )

  (:action pickup_utensil
    :parameters (?r - robot ?u - utensil ?l - region)
    :precondition (and (at_robot ?r ?l) (at_utensil ?u ?l) (handempty ?r))
    :effect (and (holding_utensil ?r ?u)
                 (not (at_utensil ?u ?l))
                 (not (handempty ?r)))
  )
 
  (:action putdown_utensil
    :parameters (?r - robot ?u - utensil ?l - region)
    :precondition (and (at_robot ?r ?l) (holding_utensil ?r ?u))
    :effect (and (at_utensil ?u ?l)
                 (handempty ?r)
                 (not (holding_utensil ?r ?u)))
  )

  (:action pickup_food
    :parameters (?r - robot ?f - food ?l - region)
    :precondition (and (at_robot ?r ?l) (at_food ?f ?l) (handempty ?r))
    :effect (and (holding_food ?r ?f)
                 (not (at_food ?f ?l))
                 (not (handempty ?r)))
  )

  (:action putdown_food
    :parameters (?r - robot ?f - food ?l - region)
    :precondition (and (at_robot ?r ?l) (holding_food ?r ?f))
    :effect (and (at_food ?f ?l)
                 (handempty ?r)
                 (not (holding_food ?r ?f)))
  )
)
