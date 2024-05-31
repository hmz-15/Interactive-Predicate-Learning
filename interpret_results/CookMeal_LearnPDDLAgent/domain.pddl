
(define (domain domain)
  (:requirements :typing)
  (:types default)
  
  (:predicates (obj_in_gripper ?v0 - default)
	(not_obj_in_gripper ?v0 - default)
	(obj_filled_with_water ?v0 - default)
	(not_obj_filled_with_water ?v0 - default)
	(obj_on_table ?v0 - default)
	(not_obj_on_table ?v0 - default)
	(obj_is_food ?v0 - default)
	(not_obj_is_food ?v0 - default)
	(obj_is_container ?v0 - default)
	(not_obj_is_container ?v0 - default)
	(obj_graspable ?v0 - default)
	(not_obj_graspable ?v0 - default)
	(gripper_empty)
	(not_gripper_empty)
	(obj_can_contain ?v0 - default ?v1 - default)
	(not_obj_can_contain ?v0 - default ?v1 - default)
	(obj_inside ?v0 - default ?v1 - default)
	(not_obj_inside ?v0 - default ?v1 - default)
	(pick_up ?v0 - default)
	(place_on_table ?v0 - default)
	(place_first_in_second ?v0 - default ?v1 - default)
	(get_water_from_faucet ?v0 - default)
	(pour_water_from_first_to_second ?v0 - default ?v1 - default)
  )
  ; (:actions pick_up place_on_table place_first_in_second get_water_from_faucet pour_water_from_first_to_second)

  

	(:action place_on_table_1
		:parameters (?v_0 - default)
		:precondition (and (obj_in_gripper ?v_0)
			(place_on_table ?v_0))
		:effect (and
			(obj_on_table ?v_0)
			(not (obj_in_gripper ?v_0))
			(gripper_empty)
			(not_obj_in_gripper ?v_0))
	)
	

	(:action pick_up_1
		:parameters (?v_0 - default)
		:precondition (and (obj_graspable ?v_0)
			(gripper_empty)
			(pick_up ?v_0))
		:effect (and
			(not (obj_on_table ?v_0))
			(not (not_obj_in_gripper ?v_0))
			(obj_in_gripper ?v_0)
			(not (gripper_empty)))
	)
	

	(:action get_water_from_faucet_1
		:parameters (?v_0 - default)
		:precondition (and (obj_is_container ?v_0)
			(obj_in_gripper ?v_0)
			(not_obj_filled_with_water ?v_0)
			(get_water_from_faucet ?v_0))
		:effect (and
			(not (not_obj_filled_with_water ?v_0))
			(obj_filled_with_water ?v_0))
	)
	

	(:action place_first_in_second_1
		:parameters (?v_1 - default ?v_0 - default)
		:precondition (and (obj_can_contain ?v_1 ?v_0)
			(obj_in_gripper ?v_0)
			(obj_is_food ?v_0)
			(place_first_in_second ?v_0 ?v_1))
		:effect (and
			(not (obj_in_gripper ?v_0))
			(gripper_empty)
			(not_obj_in_gripper ?v_0)
			(obj_inside ?v_0 ?v_1))
	)
	

	(:action pour_water_from_first_to_second_1
		:parameters (?v_1 - default ?v_0 - default)
		:precondition (and (obj_in_gripper ?v_0)
			(obj_is_container ?v_1)
			(obj_filled_with_water ?v_0)
			(obj_is_container ?v_0)
			(pour_water_from_first_to_second ?v_0 ?v_1))
		:effect (and
			(not (not_obj_filled_with_water ?v_1))
			(not_obj_filled_with_water ?v_0)
			(obj_filled_with_water ?v_1)
			(not (obj_filled_with_water ?v_0)))
	)

  

)
        