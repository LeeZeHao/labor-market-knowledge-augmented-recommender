#!/usr/bin/env bash
set -euo pipefail

cd /home/leeze/career_experiments_2

MODELS=(DIN DIEN DCNv1 DCNv2 DeepFM)
ADAPTORS=(none kar_base kar_ours)
TASKS=(company role)

for model in "${MODELS[@]}"; do
    for adaptor in "${ADAPTORS[@]}"; do
        for task in "${TASKS[@]}"; do
            echo "Training model=${model}, adaptor=${adaptor}, task=${task}"

            python main_ctr.py \
                --model "${model}" \
                --adaptor "${adaptor}" \
                --prediction_task "${task}" \
                --epoch_num 20 \
                --patience 3 \
                --save_checkpoint \
                "$@"
        done
    done
done

echo "All 30 training runs completed."