#!/bin/zsh

# world=StoreObjects
# load_path=saved_runs/exp_results/StoreObjects/train/StoreObjects_LearnPDDLAgent_1
world=SetTable
# world=CookMeal
agent=LearnPDDLAgent
# agent=OraclePlanAgent

# run render for one time
render=false

# random seeds
seeds=(42)
# seeds=(42 1319 6005)

save_dirs=()

#############################################
echo "Run train for $world"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_train/$now/${world}_pddl

for seed in "${seeds[@]}"; do
    save_dir=$save_base_dir/$seed
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_single_mode mode=train world=$world agent=$agent render=$render seed=$seed save_dir=$save_dir
done


#print out the save directories
echo "#############################################"
echo "Done training for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done