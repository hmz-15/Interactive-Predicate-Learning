#!/bin/zsh

load_paths=()

# world=StoreObjects
# load_paths+=saved_runs/exp_results/StoreObjects/train/StoreObjects_pddl/42
# load_paths+=saved_runs/exp_results/StoreObjects/train/StoreObjects_pddl/1319
# load_paths+=saved_runs/exp_results/StoreObjects/train/StoreObjects_pddl/6005

# world=SetTable
# load_paths+=saved_runs/exp_results/SetTable/train/SetTable_pddl/42
# load_paths+=saved_runs/exp_results/SetTable/train/SetTable_pddl/1319
# load_paths+=saved_runs/exp_results/SetTable/train/SetTable_pddl/6005

# bootstrap
# load_paths+=saved_runs/exp_results/SetTable/train/SetTable_pddl_bootstrapped/42
# load_paths+=saved_runs/exp_results/SetTable/train/SetTable_pddl_bootstrapped/1319
# load_paths+=saved_runs/exp_results/SetTable/train/SetTable_pddl_bootstrapped/6005

# world=CookMeal
# load_paths+=saved_runs/exp_results/CookMeal/train/CookMeal_pddl/42
# load_paths+=saved_runs/exp_results/CookMeal/train/CookMeal_pddl/1319
# load_paths+=saved_runs/exp_results/CookMeal/train/CookMeal_pddl/6005

# run render for one time
render=true

# random seeds
seeds=(42 1319 6005)

save_dirs=()

#############################################
echo "Run test for $world"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_test/$now/${world}_pddl

# jointly load_paths and seeds
for i in {1..${#load_paths[@]}}; do
    load_path=${load_paths[$i]}
    seed=${seeds[$i]}

    # pddl agent
    save_dir=$save_base_dir/$seed/pddl
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_single_mode mode=test world=$world agent=LearnPDDLAgent render=$render seed=$seed save_dir=$save_dir agent.load_path=$load_path

    # pddl agent (replan)
    # save_dir=$save_base_dir/$seed/pddl_replan
    # save_dirs+=$save_dir
    # python -m predicate_learning.main --config-name=run_single_mode mode=test world=$world agent=LearnPDDLAgent render=$render seed=$seed save_dir=$save_dir agent.load_path=$load_path agent.replan_at_each_step_at_test=true

    # llm agent with objects+scene
    # save_dir=$save_base_dir/$seed/llm_predicate
    # save_dirs+=$save_dir
    # python -m predicate_learning.main --config-name=run_single_mode mode=test world=$world agent=LLMPlanAgent render=$render seed=$seed agent.load_path=$load_path save_dir=$save_dir agent.use_few_shot_demos=true agent.use_both_text_and_predicates=true

    # # llm agent with objects
    # save_dir=$save_base_dir/$seed/llm_text
    # save_dirs+=$save_dir
    # python -m predicate_learning.main --config-name=run_single_mode mode=test world=$world agent=LLMPlanAgent render=$render seed=$seed agent.load_path=$load_path save_dir=$save_dir agent.use_learned_predicates=false agent.use_few_shot_demos=true

    # # llm agent with precond
    # save_dir=$save_base_dir/$seed/llm_predicate_precond
    # save_dirs+=$save_dir
    # python -m predicate_learning.main --config-name=run_single_mode mode=test world=$world agent=LLMPlanPrecondAgent render=$render seed=$seed agent.load_path=$load_path save_dir=$save_dir agent.use_few_shot_demos=true agent.use_both_text_and_predicates=true

    # code as policies
    # save_dir=$save_base_dir/$seed/code_as_policy
    # save_dirs+=$save_dir
    # python -m predicate_learning.main --config-name=run_single_mode_cap mode=test world=$world agent=CaPAgent render=$render seed=$seed save_dir=$save_dir
done


#print out the save directories
echo "#############################################"
echo "Done training for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done