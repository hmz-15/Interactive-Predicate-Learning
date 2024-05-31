#!/bin/zsh

worlds=(StoreObjects SetTable CookMeal)

# run render for one time
render=false

# random seeds
seed=42
# (42 1319 6005 7686 6766 2418 711)

save_dirs=()

#############################################
echo "Collect demos"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/collect_demos/$now

for world in "${worlds[@]}"; do
    save_dir=$save_base_dir/$world
    save_dirs+=$save_dir
    python -m predicate_learning.main --config-name=run_single_mode mode=train world=$world agent=OraclePlanAgent render=$render seed=$seed save_dir=$save_dir
done

#print out the save directories
echo "#############################################"
echo "Done collecting demos"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done