#!/bin/zsh

world=SetTable
agents=(LearnPDDLAgent LLMPlanAgent LLMPlanPrecondAgent)

# run render for one time
render=true

# random seeds
seed=42
# (42 1319 6005 7686 6766 2418 711)

save_dirs=()

#############################################
echo "Run all for $world"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_test/$now

for agent in "${agents[@]}"; do
    save_dir=$save_base_dir/${world}_${agent}
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_all world=$world agent=$agent render=$render seed=$seed save_dir=$save_dir
done

#print out the save directories
echo "#############################################"
echo "Done all for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done
