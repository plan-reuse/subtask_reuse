(define (domain kitchen)
  (:requirements :strips :typing)

  ;; ----------------
  ;; Types
  ;; ----------------
(:types
  robot
  region
  surface container stovetop - region
  movable
  kitchenware - movable
  utensil cookware - kitchenware
  ingredient - movable
  condiment liquid grocery - ingredient)


  ;; ----------------
  ;; Predicates
  ;; ----------------
  (:predicates
    (at_robot ?r - robot ?l - region)
    (at ?m - movable ?l - region)

    ;; storage / placement
    (inside ?i - ingredient ?c - container)
    (in_cookware ?g - grocery ?cw - cookware)  
    (on_utensil  ?g - grocery ?u - utensil) 

    ;; doors
    (door_open  ?c - container)
    (door_closed ?c - container)

    ;; manipulation state
    (holding ?r - robot ?m - movable)
    (handempty ?r - robot)

    ;; stove use
    (turned_on  ?s - stovetop)
    (turned_off ?s - stovetop)
    (on_stovetop ?cw - cookware ?s - stovetop)

    ;; cooking result (solids/groceries only)
    (cooked ?g - grocery)
    (cooking_finalized ?cw - cookware)

    ;; effects from pouring/sprinkling while still holding the container
    (poured    ?l - liquid   ?cw - cookware)
    (sprinkled ?c - condiment ?cw - cookware)
  )

  ;; ----------------
  ;; Navigation
  ;; ----------------
  (:action move
    :parameters (?r - robot ?from - region ?to - region)
    :precondition (and (at_robot ?r ?from))
    :effect (and (not (at_robot ?r ?from))
                 (at_robot ?r ?to))
  )

  ;; ----------------
  ;; Containers (open/close)
  ;; ----------------
  (:action open_container
    :parameters (?r - robot ?c - container)
    :precondition (and (at_robot ?r ?c)
                       (door_closed ?c)
                       (handempty ?r))
    :effect (and (door_open ?c)
                 (not (door_closed ?c)))
  )

  (:action close_container
    :parameters (?r - robot ?c - container)
    :precondition (and (at_robot ?r ?c)
                       (door_open ?c)
                       (handempty ?r))
    :effect (and (door_closed ?c)
                 (not (door_open ?c)))
  )

  ;; ----------------
  ;; Region pick/place (split by kind)
  ;; ----------------
  (:action pickup_ingredient
    :parameters (?r - robot ?i - ingredient ?l - surface)
    :precondition (and (at_robot ?r ?l)
                       (at ?i ?l)
                       (handempty ?r))
    :effect (and (holding ?r ?i)
                 (not (at ?i ?l))
                 (not (handempty ?r)))
  )

  (:action putdown_ingredient
    :parameters (?r - robot ?i - ingredient ?l - surface)
    :precondition (and (at_robot ?r ?l)
                       (holding ?r ?i))
    :effect (and (at ?i ?l)
                 (handempty ?r)
                 (not (holding ?r ?i)))
  )

  (:action pickup_kitchenware
    :parameters (?r - robot ?k - kitchenware ?l - surface)
    :precondition (and (at_robot ?r ?l)
                       (at ?k ?l)
                       (handempty ?r))
    :effect (and (holding ?r ?k)
                 (not (at ?k ?l))
                 (not (handempty ?r)))
  )

  (:action putdown_kitchenware
    :parameters (?r - robot ?k - kitchenware ?l - surface)
    :precondition (and (at_robot ?r ?l)
                       (holding ?r ?k))
    :effect (and (at ?k ?l)
                 (handempty ?r)
                 (not (holding ?r ?k)))
  )

  ;; ----------------
  ;; Containers (ingredient-only, renamed)
  ;; ----------------
  (:action retrieve_ingredient
    :parameters (?r - robot ?i - ingredient ?c - container)
    :precondition (and (at_robot ?r ?c)
                       (inside ?i ?c)
                       (door_open ?c)
                       (handempty ?r))
    :effect (and (holding ?r ?i)
                 (not (inside ?i ?c))
                 (not (handempty ?r)))
  )

  (:action store_ingredient
    :parameters (?r - robot ?i - ingredient ?c - container)
    :precondition (and (at_robot ?r ?c)
                       (holding ?r ?i)
                       (door_open ?c))
    :effect (and (inside ?i ?c)
                 (handempty ?r)
                 (not (holding ?r ?i)))
  )

  ;; ----------------
  ;; Put/Take with cookware & utensils (generalized to ingredient)
  ;; ----------------
  (:action put_ingredient_in_cookware
    :parameters (?r - robot ?g - grocery ?cw - cookware ?l - region)
    :precondition (and (at_robot ?r ?l)
                       (at ?cw ?l)
                       (holding ?r ?g))
    :effect (and (in_cookware ?g ?cw)
                 (handempty ?r)
                 (not (holding ?r ?g)))
  )

  (:action take_ingredient_from_cookware
    :parameters (?r - robot ?g - grocery ?cw - cookware ?l - region)
    :precondition (and (at_robot ?r ?l)
                       (at ?cw ?l)
                       (in_cookware ?g ?cw)
                       (handempty ?r))
    :effect (and (holding ?r ?g)
                 (not (in_cookware ?g ?cw))
                 (not (handempty ?r)))
  )

  (:action place_ingredient_on_utensil
    :parameters (?r - robot ?g - grocery ?u - utensil ?l - region)
    :precondition (and (at_robot ?r ?l)
                       (at ?u ?l)
                       (holding ?r ?g))
    :effect (and (on_utensil ?g ?u)
                 (handempty ?r)
                 (not (holding ?r ?g)))
  )

  (:action take_ingredient_from_utensil
    :parameters (?r - robot ?g - grocery ?u - utensil ?l - region)
    :precondition (and (at_robot ?r ?l)
                       (at ?u ?l)
                       (on_utensil ?g ?u)
                       (handempty ?r))
    :effect (and (holding ?r ?g)
                 (not (on_utensil ?g ?u))
                 (not (handempty ?r)))
  )

  ;; ----------------
  ;; Stove and cookware placement
  ;; ----------------
  (:action place_cookware_on_stovetop
    :parameters (?r - robot ?cw - cookware ?s - stovetop)
    :precondition (and (at_robot ?r ?s)
                       (holding ?r ?cw))
    :effect (and (on_stovetop ?cw ?s)
                 (handempty ?r)
                 (not (holding ?r ?cw)))
  )

  (:action remove_cookware_from_stovetop
    :parameters (?r - robot ?cw - cookware ?s - stovetop)
    :precondition (and (at_robot ?r ?s)
                       (on_stovetop ?cw ?s)
                       (handempty ?r))
    :effect (and (holding ?r ?cw)
                 (not (on_stovetop ?cw ?s))
                 (not (handempty ?r)))
  )

  (:action turn_on_stove
    :parameters (?r - robot ?s - stovetop)
    :precondition (and (at_robot ?r ?s)
                       (turned_off ?s)
                       (handempty ?r))
    :effect (and (turned_on ?s)
                 (not (turned_off ?s)))
  )

  (:action turn_off_stove
    :parameters (?r - robot ?s - stovetop)
    :precondition (and (at_robot ?r ?s)
                       (turned_on ?s)
                       (handempty ?r))
    :effect (and (turned_off ?s)
                 (not (turned_on ?s)))
  )

  ;; ----------------
  ;; New actions: pouring & sprinkling while still holding the container
  ;; ----------------
(:action pour_from_bottle
  :parameters (?r - robot ?l - liquid ?cw - cookware ?lcn - region)
  :precondition (and (at_robot ?r ?lcn)
                     (at ?cw ?lcn)
                     (holding ?r ?l)
                     (not (cooking_finalized ?cw)))
  :effect (and (poured ?l ?cw))
)

(:action sprinkle_condiment
  :parameters (?r - robot ?c - condiment ?cw - cookware ?lcn - region)
  :precondition (and (at_robot ?r ?lcn)
                     (at ?cw ?lcn)
                     (holding ?r ?c)
                     (not (cooking_finalized ?cw)))
  :effect (and (sprinkled ?c ?cw))
)


  ;; ----------------
  ;; Cooking progression (groceries)
  ;; ----------------
(:action cooking_food
  :parameters (?s - stovetop ?cw - cookware ?g - grocery)
  :precondition (and (in_cookware ?g ?cw)
                     (on_stovetop ?cw ?s)
                     (turned_on ?s))
  :effect (and (cooked ?g)
               (cooking_finalized ?cw))
)

)

