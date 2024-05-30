#!/bin/zsh

# world=StoreObjects
load_path=saved_runs/run_real_train/2024-02-08-09-13-01/StoreObjects_pddl/train/7
world=SetTable
agent=LearnPDDLAgent
# agent=OraclePlanAgent

reuse=true

# random seeds
seed=52
# (42 1319 6005 7686 6766 2418 711)

save_dirs=()

#############################################
echo "Run train for $world"
echo "[Please press enter to continue]"
read space

now=$(/bin/date +%Y-%m-%d-%H-%M-%S)
save_base_dir=saved_runs/run_real_train/$now/${world}_pddl

# jointly load_paths and seeds
save_dir=$save_base_dir
save_dirs+=$save_dir
python -m real_robot.main --config-name=run_train world=$world agent=$agent save_dir=$save_dir seed=$seed agent.load_path=$load_path load_agent_knowledge=$reuse visualize_obj_pc=True use_template_feedback=True record_video=True overwrite_instances=true

#print out the save directories
echo "#############################################"
echo "Done training for $world"
echo "Save to:"
for save_dir in "${save_dirs[@]}"; do
  echo "- $save_dir"
done