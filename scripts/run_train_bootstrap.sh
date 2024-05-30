#!/bin/zsh

# world=StoreObjects
# load_path=saved_runs/exp_results/StoreObjects/train/StoreObjects_LearnPDDLAgent_1
world=SetTable
# world=CookMeal
agent=LearnPDDLAgent
# agent=OraclePlanAgent

load_paths=()
load_paths+=saved_runs/exp_results/StoreObjects/train/StoreObjects_LearnPDDLAgent_1
load_paths+=saved_runs/run_train/2024-01-30-14-25-38/StoreObjects_pddl/1319
load_paths+=saved_runs/run_train/2024-01-31-00-29-17/StoreObjects_pddl/6005

reuse=true

# run render for one time
render=true

# random seeds
seeds=(42 1319 6005)
# (42 1319 6005 7686 6766 2418 711)

save_dirs=()

#############################################
echo "Run train for $world"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_train/$now/${world}_pddl

# jointly load_paths and seeds
for i in {1..${#load_paths[@]}}; do
    load_path=${load_paths[$i]}
    seed=${seeds[$i]}

    save_dir=$save_base_dir/$seed
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_single_mode mode=train world=$world agent=$agent render=$render seed=$seed save_dir=$save_dir agent.load_path=$load_path load_agent_knowledge=$reuse
done

#print out the save directories
echo "#############################################"
echo "Done training for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done