#!/bin/zsh

world=$1
# world=StackBlocks
agent=$2
num_trails=3

# run render for one time
render=true

# random seeds
seeds=(42 1319 6005 7686 6766 2418 711)

save_dirs=()

#############################################
echo "Run $agent for $world"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_all/$now/${world}_$agent
for index in {1..$num_trails}; do
    save_dir=$save_base_dir/$index
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_all world=$world agent=$agent seed=${seeds[index]} save_dir=$save_dir render=$render
    render=false
done

#print out the save directories
echo "#############################################"
echo "Done training for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done