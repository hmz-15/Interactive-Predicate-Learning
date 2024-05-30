(define (domain cook_meal)  

  (:predicates (on_table ?v)
	(inside_container ?v0 ?v1)
	(in_gripper ?v)
	(is_graspable ?v)
	(is_container ?v)
	(is_food ?v)
	(has_water ?v)
	(has_no_water ?v)
	(not_in_container ?v)
	(can_contain_food ?v)
	(gripper_empty )
	(pick_up ?v)
	(place_on_table ?v)
	(place_first_in_second ?v0 ?v1)
	(get_water_from_faucet ?v)
	(pour_water_from_first_to_second ?v0 ?v1)
  )
  ; (:actions pick_up place_on_table place_first_in_second get_water_from_faucet pour_water_from_first_to_second)


	(:action pick-from-table
		:parameters (?o)
		:precondition (and (pick_up ?o)
			(gripper_empty )
			(not_in_container ?o)
			(is_graspable ?o))
		:effect (and
			(in_gripper ?o)
			(not (on_table ?o))
			(not (gripper_empty )))
	)


	(:action place-on-table
		:parameters (?o)
		:precondition (and (place_on_table ?o)
			(in_gripper ?o))
		:effect (and
			(gripper_empty )
			(not (in_gripper ?o))
			(on_table ?o))
	)


	(:action place-in-container
		:parameters (?o1 ?o2)
		:precondition (and (place_first_in_second ?o1 ?o2)
			(in_gripper ?o1)
			(is_container ?o2)
			(is_food ?o1)
			(can_contain_food ?o2)
			(not_in_container ?o1))
		:effect (and
			(inside_container ?o1 ?o2)
			(gripper_empty )
			(not (in_gripper ?o1))
			(not (not_in_container ?o1)))
	)


	(:action get-water-from-faucet
		:parameters (?o)
		:precondition (and (get_water_from_faucet ?o)
			(in_gripper ?o)
			(is_container ?o)
			(not_in_container ?o)
			(has_no_water ?o))
		:effect (and
			(has_water ?o)
			(not (has_no_water ?o)))
	)


	(:action pour-water-into-container
		:parameters (?o1 ?o2)
		:precondition (and (pour_water_from_first_to_second ?o1 ?o2)
			(in_gripper ?o1)
			(has_water ?o1)
			(is_container ?o2)
			(not_in_container ?o1)
			(not_in_container ?o2))
		:effect (and
			(has_water ?o2)
			(has_no_water ?o1)
			(not (has_water ?o1)))
	)
)