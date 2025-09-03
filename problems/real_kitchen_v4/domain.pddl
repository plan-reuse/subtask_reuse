(define (domain kitchen)
  (:requirements :strips :typing)
  
  (:types 
    robot 
    region 
    container - region
    appliance - container
    item
    utensil - item
    food - item
  )
  
  (:predicates
    (at_robot ?r - robot ?l - region)
    (at ?i - item ?l - region)
    (inside ?i - item ?c - container)
    (door_open ?c - container)
    (door_closed ?c - container)
    (holding ?r - robot ?i - item)
    (handempty ?r - robot)
    (turned_on ?a - appliance)    
    (turned_off ?a - appliance)  
    (cooked ?f - food)           
  )
  
  (:action move
    :parameters (?r - robot ?from - region ?to - region)
    :precondition (and (at_robot ?r ?from))
    :effect (and (not (at_robot ?r ?from)) (at_robot ?r ?to))
  )
  
  (:action open_container
    :parameters (?r - robot ?c - container)
    :precondition (and 
                    (at_robot ?r ?c)
                    (door_closed ?c)
                    (handempty ?r))
    :effect (and 
              (door_open ?c)
              (not (door_closed ?c)))
  )
  
  (:action close_container
    :parameters (?r - robot ?c - container)
    :precondition (and 
                    (at_robot ?r ?c)
                    (door_open ?c)
                    (handempty ?r))
    :effect (and 
              (door_closed ?c)
              (not (door_open ?c)))
  )
  
  (:action pickup_from_region
    :parameters (?r - robot ?i - item ?l - region)
    :precondition (and 
                    (at_robot ?r ?l) 
                    (at ?i ?l) 
                    (handempty ?r))
    :effect (and 
              (holding ?r ?i)
              (not (at ?i ?l))
              (not (handempty ?r)))
  )
  
  (:action putdown_to_region
    :parameters (?r - robot ?i - item ?l - region)
    :precondition (and 
                    (at_robot ?r ?l) 
                    (holding ?r ?i))
    :effect (and 
              (at ?i ?l)
              (handempty ?r)
              (not (holding ?r ?i)))
  )
  
  (:action pickup_from_container
    :parameters (?r - robot ?i - item ?c - container)
    :precondition (and 
                    (at_robot ?r ?c)
                    (inside ?i ?c)
                    (door_open ?c)
                    (handempty ?r))
    :effect (and 
              (holding ?r ?i)
              (not (inside ?i ?c))
              (not (handempty ?r)))
  )
  
  (:action putdown_to_container
    :parameters (?r - robot ?i - item ?c - container)
    :precondition (and 
                    (at_robot ?r ?c)
                    (holding ?r ?i)
                    (door_open ?c))
    :effect (and 
              (inside ?i ?c)
              (handempty ?r)
              (not (holding ?r ?i)))
  )
  
  (:action turn_on_appliance
    :parameters (?r - robot ?a - appliance)
    :precondition (and 
                    (at_robot ?r ?a)
                    (door_closed ?a)
                    (turned_off ?a)
                    (handempty ?r))
    :effect (and 
              (turned_on ?a)
              (not (turned_off ?a)))
  )
  
  (:action turn_off_appliance
    :parameters (?r - robot ?a - appliance)
    :precondition (and 
                    (at_robot ?r ?a)
                    (turned_on ?a)
                    (handempty ?r))
    :effect (and 
              (turned_off ?a)
              (not (turned_on ?a)))
  )
  
  (:action cook_food
    :parameters (?a - appliance ?f - food)
    :precondition (and 
                    (inside ?f ?a)
                    (door_closed ?a)
                    (turned_on ?a))
    :effect (and 
              (cooked ?f))
  )
)

