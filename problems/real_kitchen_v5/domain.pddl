(define (domain kitchen)
  (:requirements :strips :typing)
  
  (:types 
    robot 
    region 
    container - region
    stovetop - region
    movable
    utensil - movable
    cookware - movable
    food - movable
  )
  
  (:predicates
    (at_robot ?r - robot ?l - region)
    (at ?m - movable ?l - region)
    (inside ?m - movable ?c - container)
    (in_cookware ?f - food ?cw - cookware)
    (on_utensil ?f - food ?u - utensil)
    (door_open ?c - container)
    (door_closed ?c - container)
    (holding ?r - robot ?m - movable)
    (handempty ?r - robot)
    (turned_on ?s - stovetop)    
    (turned_off ?s - stovetop)
    (on_stovetop ?cw - cookware ?s - stovetop)  
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
    :parameters (?r - robot ?m - movable ?l - region)
    :precondition (and 
                    (at_robot ?r ?l) 
                    (at ?m ?l) 
                    (handempty ?r))
    :effect (and 
              (holding ?r ?m)
              (not (at ?m ?l))
              (not (handempty ?r)))
  )
  
  (:action putdown_to_region
    :parameters (?r - robot ?m - movable ?l - region)
    :precondition (and 
                    (at_robot ?r ?l) 
                    (holding ?r ?m))
    :effect (and 
              (at ?m ?l)
              (handempty ?r)
              (not (holding ?r ?m)))
  )
  
  (:action pickup_from_container
    :parameters (?r - robot ?m - movable ?c - container)
    :precondition (and 
                    (at_robot ?r ?c)
                    (inside ?m ?c)
                    (door_open ?c)
                    (handempty ?r))
    :effect (and 
              (holding ?r ?m)
              (not (inside ?m ?c))
              (not (handempty ?r)))
  )
  
  (:action putdown_to_container
    :parameters (?r - robot ?m - movable ?c - container)
    :precondition (and 
                    (at_robot ?r ?c)
                    (holding ?r ?m)
                    (door_open ?c))
    :effect (and 
              (inside ?m ?c)
              (handempty ?r)
              (not (holding ?r ?m)))
  )
  
  (:action put_food_in_cookware
    :parameters (?r - robot ?f - food ?cw - cookware ?l - region)
    :precondition (and 
                    (at_robot ?r ?l)
                    (at ?cw ?l)
                    (holding ?r ?f))
    :effect (and 
              (in_cookware ?f ?cw)
              (handempty ?r)
              (not (holding ?r ?f)))
  )
  
  
  (:action take_food_from_cookware
    :parameters (?r - robot ?f - food ?cw - cookware ?l - region)
    :precondition (and 
                    (at_robot ?r ?l)
                    (at ?cw ?l)
                    (in_cookware ?f ?cw)
                    (handempty ?r))
    :effect (and 
              (holding ?r ?f)
              (not (in_cookware ?f ?cw))
              (not (handempty ?r)))
  )
  
  (:action place_food_on_utensil
    :parameters (?r - robot ?f - food ?u - utensil ?l - region)
    :precondition (and 
                    (at_robot ?r ?l)
                    (at ?u ?l)
                    (holding ?r ?f))
    :effect (and 
              (on_utensil ?f ?u)
              (handempty ?r)
              (not (holding ?r ?f)))
  )
  
  (:action take_food_from_utensil
    :parameters (?r - robot ?f - food ?u - utensil ?l - region)
    :precondition (and 
                    (at_robot ?r ?l)
                    (at ?u ?l)
                    (on_utensil ?f ?u)
                    (handempty ?r))
    :effect (and 
              (holding ?r ?f)
              (not (on_utensil ?f ?u))
              (not (handempty ?r)))
  )
  
  (:action place_cookware_on_stovetop
    :parameters (?r - robot ?cw - cookware ?s - stovetop)
    :precondition (and 
                    (at_robot ?r ?s)
                    (holding ?r ?cw))
    :effect (and 
              (on_stovetop ?cw ?s)
              (handempty ?r)
              (not (holding ?r ?cw)))
  )
  
  (:action remove_cookware_from_stovetop
    :parameters (?r - robot ?cw - cookware ?s - stovetop)
    :precondition (and 
                    (at_robot ?r ?s)
                    (on_stovetop ?cw ?s)
                    (handempty ?r))
    :effect (and 
              (holding ?r ?cw)
              (not (on_stovetop ?cw ?s))
              (not (handempty ?r)))
  )
  
  (:action turn_on_stove
    :parameters (?r - robot ?s - stovetop)
    :precondition (and 
                    (at_robot ?r ?s)
                    (turned_off ?s)
                    (handempty ?r))
    :effect (and 
              (turned_on ?s)
              (not (turned_off ?s)))
  )
  
  (:action turn_off_stove
    :parameters (?r - robot ?s - stovetop)
    :precondition (and 
                    (at_robot ?r ?s)
                    (turned_on ?s)
                    (handempty ?r))
    :effect (and 
              (turned_off ?s)
              (not (turned_on ?s)))
  )
  
  (:action wait_for_food_to_cook
    :parameters (?s - stovetop ?cw - cookware ?f - food)
    :precondition (and 
                    (in_cookware ?f ?cw)
                    (on_stovetop ?cw ?s)
                    (turned_on ?s))
    :effect (and 
              (cooked ?f))
  )
)

