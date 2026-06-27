#!/bin/bash
# Launch the LIFT3-matched FastSAC baseline (IsaacLab port) in the lift3:bm-wbt
# image. Trains holosoma's FastSAC algorithm on the SAME IsaacLab G1 tracking env
# as the PPO baseline (run_lift_match_ppo.sh) -> only the algorithm differs.
# Logs eval/* to the shared fight1_baselines wandb project for direct comparison.
set -euo pipefail

GPU=${GPU:-5}
SEED=${SEED:-1}
EXP_TAG=${EXP_TAG:-fight1_fastsac_isaaclab}
WBT_DIR=/home/weidong/whole_body_tracking
ISAACLAB_LOCAL=/home/weidong/IsaacLab_localcopy
MOTION=${MOTION:-/home/weidong/LIFT3_wt_simba_mlphead/tracking_motion/motion_fight1_subject2_cut2_mujoco.npz}
MOTION_DIR=$(dirname "${MOTION}")
WANDB_PROJECT=${WANDB_PROJECT:-fight1_baselines}
WANDB_API_KEY=${WANDB_API_KEY:-ce601da9131d4839740cb8da8c4f34aaa2e74ee8}
MAX_ITER=${MAX_ITER:-400000}
NUM_ENVS=${NUM_ENVS:-1000}

TS=$(date +%Y%m%d-%H%M%S)
RUN_DIR=${WBT_DIR}/sac_lift_match_runs/${TS}-${EXP_TAG}
mkdir -p "${RUN_DIR}"
chmod -R 777 "${RUN_DIR}" || true
mkdir -p /home/weidong/.cache/bm_wbt_ov_sac_${EXP_TAG}

CONT=lift3-bm-sac-${EXP_TAG}
docker rm -f "${CONT}" 2>/dev/null || true

echo "[sac] EXP_TAG=${EXP_TAG} GPU=${GPU} seed=${SEED}"
echo "[sac] motion=${MOTION}"
echo "[sac] RUN_DIR=${RUN_DIR}"

docker run -d --name "${CONT}" \
  --runtime=nvidia --gpus "\"device=${GPU}\"" \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 \
  -e ISAACLAB_PATH=${ISAACLAB_LOCAL} \
  -e PYTHONPATH=${WBT_DIR}/source/whole_body_tracking \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -e WANDB_USERNAME=bigeasthuang \
  -e WANDB_ENTITY=bigeasthuang \
  -e WANDB_PROJECT=${WANDB_PROJECT} \
  -e WANDB_RUN_NAME=fight1_fastsac_isaaclab_s${SEED} \
  -e WANDB_MODE=${WANDB_MODE:-online} \
  -e WANDB_DISABLE_GIT=true -e WANDB_DISABLE_CODE=true \
  -v ${WBT_DIR}:${WBT_DIR} \
  -v ${ISAACLAB_LOCAL}:${ISAACLAB_LOCAL} \
  -v ${MOTION_DIR}:${MOTION_DIR}:ro \
  -v /home/weidong/.cache/bm_wbt_ov_sac_${EXP_TAG}:/root/.cache/ov \
  --entrypoint bash lift3:bm-wbt -lc "
    git config --global --add safe.directory '*' || true
    cd ${WBT_DIR}/scripts/rsl_rl
    /isaac-sim/python.sh train_sac_match.py \
      --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
      --motion_file ${MOTION} \
      --num_envs ${NUM_ENVS} \
      --seed ${SEED} \
      --max_iterations ${MAX_ITER} \
      --headless --device cuda:0 \
      > ${RUN_DIR}/train.log 2>&1
  "
echo "[sac] container ${CONT} started"
echo "[sac] tail -f ${RUN_DIR}/train.log"
echo "${RUN_DIR}" > ${WBT_DIR}/sac_lift_match_runs/LATEST_${EXP_TAG}.txt
