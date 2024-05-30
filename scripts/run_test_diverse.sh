#!/bin/zsh

load_paths=()
world=StoreObjects
load_paths+=saved_runs/run_train/2024-02-07-15-01-31/StoreObjects_pddl/1
load_paths+=saved_runs/run_train/2024-02-07-15-01-31/StoreObjects_pddl/2
load_paths+=saved_runs/run_train/2024-02-07-15-01-31/StoreObjects_pddl/3

# run render for one time
render=false

# random seeds
seed=42

save_dirs=()

#############################################
echo "Run test diverse feedback for $world"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_test/$now/${world}_pddl

# jointly load_paths and seeds
for i in {1..${#load_paths[@]}}; do
    load_path=${load_paths[$i]}

    # pddl agent
    save_dir=$save_base_dir/$seed_$i
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_single_mode mode=test world=$world agent=LearnPDDLAgent render=$render seed=$seed save_dir=$save_dir agent.load_path=$load_path
done


#print out the save directories
echo "#############################################"
echo "Done training for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done