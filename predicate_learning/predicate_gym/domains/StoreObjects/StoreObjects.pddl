(define (domain store_object)  

  (:predicates (on_table ?v)
	(on ?v0 ?v1)
	(in_gripper ?v)
	(clear_on_top ?v)
	(is_graspable ?v)
	(gripper_empty  )
	(pick_up ?v)
	(place_on_table ?v)
	(place_first_on_second ?v0 ?v1)
  )
  ; (:actions pick_up place_on_table place_first_on_second)


	(:action pick-from-table
		:parameters (?o)
		:precondition (and (pick_up ?o)
			(gripper_empty )
			(is_graspable ?o)
			(clear_on_top ?o)
			(on_table ?o))
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


	(:action pick-from-object
		:parameters (?o1 ?o2)
		:precondition (and (pick_up ?o1)
			(gripper_empty )
			(is_graspable ?o1)
			(clear_on_top ?o1)
			(on ?o1 ?o2))
		:effect (and
			(in_gripper ?o1)
			(not (on ?o1 ?o2))
			(not (gripper_empty ))
			(clear_on_top ?o2))
	)


	(:action place-on-object
		:parameters (?o1 ?o2)
		:precondition (and (place_first_on_second ?o1 ?o2)
			(in_gripper ?o1)
			(clear_on_top ?o2))
		:effect (and
			(not (in_gripper ?o1))
			(gripper_empty )
			(on ?o1 ?o2)
			(not (clear_on_top ?o2)))
	)

  

)
        