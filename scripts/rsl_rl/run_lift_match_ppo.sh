#!/bin/bash
# Launch the LIFT3-matched whole_body_control PPO baseline (GPU6) in the
# lift3:bm-wbt image (IsaacSim4.5 + IsaacLab2.1 + rsl_rl). Mirrors the bm_wbt
# collector's docker invocation but runs scripts/rsl_rl/train_lift_match.py.
#
#   * env reward/physics matched to the GPU0-5 SAC (undesired_contacts OFF),
#   * NO domain randomization, adaptive motion sampling ON, terminations default,
#   * num_envs=1000, num_steps_per_env=20, save_interval=50,
#   * logs Metrics/avg_total_reward + Metrics/avg_episode_length per 1000 steps.
set -euo pipefail

GPU=${GPU:-6}
SEED=${SEED:-1}
EXP_TAG=${EXP_TAG:-fight1_ppo_lift_match}
WBT_DIR=/home/weidong/whole_body_tracking
ISAACLAB_LOCAL=/home/weidong/IsaacLab_localcopy
MOTION=${MOTION:-/home/weidong/LIFT3_wt_wm_cem/tracking_motion/motion_fight1_subject2_cut2_mujoco.npz}
MOTION_DIR=$(dirname "${MOTION}")
WANDB_PROJECT=${WANDB_PROJECT:-fight1_ppo_lift_match}
WANDB_API_KEY=${WANDB_API_KEY:-ce601da9131d4839740cb8da8c4f34aaa2e74ee8}
MAX_ITER=${MAX_ITER:-30000}
NUM_ENVS=${NUM_ENVS:-1000}

TS=$(date +%Y%m%d-%H%M%S)
RUN_DIR=${WBT_DIR}/ppo_lift_match_runs/${TS}-${EXP_TAG}
mkdir -p "${RUN_DIR}"
chmod -R 777 "${RUN_DIR}" || true
mkdir -p /home/weidong/.cache/bm_wbt_ov_ppo_${EXP_TAG}

CONT=lift3-bm-ppo-${EXP_TAG}
docker rm -f "${CONT}" 2>/dev/null || true

echo "[ppo] EXP_TAG=${EXP_TAG} GPU=${GPU} seed=${SEED}"
echo "[ppo] motion=${MOTION}"
echo "[ppo] RUN_DIR=${RUN_DIR}"

docker run -d --name "${CONT}" \
  --runtime=nvidia --gpus "\"device=${GPU}\"" \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 \
  -e ISAACLAB_PATH=${ISAACLAB_LOCAL} \
  -e PYTHONPATH=${WBT_DIR}/source/whole_body_tracking \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -e WANDB_USERNAME=bigeasthuang \
  -v ${WBT_DIR}:${WBT_DIR} \
  -v ${ISAACLAB_LOCAL}:${ISAACLAB_LOCAL} \
  -v ${MOTION_DIR}:${MOTION_DIR}:ro \
  -v /home/weidong/.cache/bm_wbt_ov_ppo_${EXP_TAG}:/root/.cache/ov \
  --entrypoint bash lift3:bm-wbt -lc "
    # container runs as root over weidong-owned mounts: let git (used by wandb's
    # repo capture + runner.add_git_repo_to_log) trust them, else wandb.init dies.
    git config --global --add safe.directory '*' || true
    cd ${WBT_DIR}/scripts/rsl_rl
    /isaac-sim/python.sh train_lift_match.py \
      --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
      --motion_file ${MOTION} \
      --num_envs ${NUM_ENVS} \
      --seed ${SEED} \
      --max_iterations ${MAX_ITER} \
      --window_steps 1000 \
      --headless --device cuda:0 \
      --logger wandb --log_project_name ${WANDB_PROJECT} \
      --run_name ${EXP_TAG}_s${SEED} \
      > ${RUN_DIR}/train.log 2>&1
  "
echo "[ppo] container ${CONT} started"
echo "[ppo] tail -f ${RUN_DIR}/train.log"
echo "${RUN_DIR}" > ${WBT_DIR}/ppo_lift_match_runs/LATEST_${EXP_TAG}.txt
