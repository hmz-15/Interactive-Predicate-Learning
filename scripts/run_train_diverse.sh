#!/bin/zsh

world=StoreObjects
agent=LearnPDDLAgent

# run render for one time
render=false

# run trails
num_trails=3

# random seeds
seed=42
# seeds=(42 1319 6005)
# (42 1319 6005 7686 6766 2418 711)

save_dirs=()

#############################################
echo "Run train with diverse feedback for $world with $num_trails trails"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_train/$now/${world}_pddl

# iterate for trails
for index in {1..$num_trails}; do
    save_dir=$save_base_dir/$seed_$index
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_single_mode mode=train world=$world agent=$agent render=$render seed=$seed save_dir=$save_dir use_diverse_feedback=true
done


#print out the save directories
echo "#############################################"
echo "Done training for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done