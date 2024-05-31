
(define (domain domain)
  (:requirements :typing)
  (:types default)
  
  (:predicates (obj_in_gripper ?v0 - default)
	(not_obj_in_gripper ?v0 - default)
	(obj_on_obj ?v0 - default ?v1 - default)
	(not_obj_on_obj ?v0 - default ?v1 - default)
	(graspable ?v0 - default)
	(not_graspable ?v0 - default)
	(obj_on_table ?v0 - default)
	(not_obj_on_table ?v0 - default)
	(gripper_empty)
	(not_gripper_empty)
	(obj_clear ?v0 - default)
	(not_obj_clear ?v0 - default)
	(obj_thin_enough ?v0 - default)
	(not_obj_thin_enough ?v0 - default)
	(is_plate ?v0 - default)
	(not_is_plate ?v0 - default)
	(pick_up ?v0 - default)
	(place_on_table ?v0 - default)
	(place_first_on_second ?v0 - default ?v1 - default)
	(push_plate_on_object ?v0 - default ?v1 - default)
  )
  ; (:actions pick_up place_on_table place_first_on_second push_plate_on_object)

  

	(:action pick_up_1
		:parameters (?v_0 - default ?v_2 - default)
		:precondition (and (gripper_empty)
			(obj_clear ?v_0)
			(obj_on_obj ?v_0 ?v_2)
			(graspable ?v_0)
			(pick_up ?v_0))
		:effect (and
			(not_obj_on_obj ?v_0 ?v_2)
			(obj_clear ?v_2)
			(not (gripper_empty))
			(obj_in_gripper ?v_0)
			(not (obj_on_obj ?v_0 ?v_2)))
	)
	

	(:action pick_up_2
		:parameters (?v_0 - default)
		:precondition (and (gripper_empty)
			(obj_clear ?v_0)
			(graspable ?v_0)
			(obj_on_table ?v_0)
			(pick_up ?v_0))
		:effect (and
			(not (obj_on_table ?v_0))
			(not (gripper_empty))
			(obj_in_gripper ?v_0))
	)
	

	(:action place_first_on_second_1
		:parameters (?v_0 - default ?v_1 - default)
		:precondition (and (obj_in_gripper ?v_0)
			(obj_clear ?v_1)
			(place_first_on_second ?v_0 ?v_1))
		:effect (and
			(not (obj_clear ?v_1))
			(not (not_obj_on_obj ?v_0 ?v_1))
			(gripper_empty)
			(not (obj_in_gripper ?v_0))
			(obj_on_obj ?v_0 ?v_1))
	)
	

	(:action place_on_table_1
		:parameters (?v_0 - default)
		:precondition (and (obj_in_gripper ?v_0)
			(place_on_table ?v_0))
		:effect (and
			(gripper_empty)
			(not (obj_in_gripper ?v_0))
			(obj_on_table ?v_0))
	)
	

	(:action push_plate_on_object_1
		:parameters (?v_0 - default ?v_1 - default)
		:precondition (and (gripper_empty)
			(obj_clear ?v_0)
			(obj_thin_enough ?v_1)
			(is_plate ?v_0)
			(obj_clear ?v_1)
			(push_plate_on_object ?v_0 ?v_1))
		:effect (and
			(not (obj_on_table ?v_0))
			(not (obj_clear ?v_1))
			(obj_on_obj ?v_0 ?v_1)
			(not (not_obj_on_obj ?v_0 ?v_1)))
	)

  

)
        