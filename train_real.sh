#!/bin/bash

declare -a old_scenes=( \
    "data_color25" \
)

declare -a new_scenes=( \
    "pillow" \
    "fan" \
    "baseball" \
    "target1" \
    "jacks1" \
)

target_scene=$1
lrate_base=0.00005
acc_factor=4
flow_prior_weight=0.01
batch_size=$((256 * acc_factor))
lrate=$(echo "$lrate_base * $acc_factor" | bc)
frames=60
additional_args=$2

for scene_type in new old; do
    tof_multiplier=1
    if [ $scene_type == "new" ]; then
        echo "Processing new scenes"
        scenes=("${new_scenes[@]}")
        script="./scripts/run_real.sh"
        warped_color_weight=0.0
        if [ "$target_scene" == "pillow" ]; then
            tof_multiplier=2
        fi
    else
        echo "Processing old scenes"
        scenes=("${old_scenes[@]}")
        script="./scripts/run_real_legacy.sh"
        warped_color_weight=0.01
    fi
    for scene in "${scenes[@]}"; do
        if [ "$target_scene" != "" ] && [ "$target_scene" != "$scene" ]; then
            echo "Skipping scene $scene, not matching target scene $target_scene"
            continue
        fi
        echo "Processing $scene"
        
        ./$script $frames $scene $scene \
                    "--dc_offset 0.0 --pretraining_scheme stages --lrate_decay 1000 --lrate_decay_calib 1000 --N_iters 400001 --tof_loss_norm L2 \
                    --pretraining_stage1_iters 25000 --pretraining_stage2_iters 1000000 --tof_multiplier ${tof_multiplier} --temporal_embedding PE \
                    --quad_multiplier 0.5 --color_weight 0.0 --N_rand 256 --i_video 25000 --i_img 10000 \
                    --lrate ${lrate} --lrate_calib ${lrate} --gradient_accumulation_factor ${acc_factor} --resume \
                    --scene_flow_weight_decay_steps 10000 --tof_weight_decay_steps 0 --sparsity_weight_decay_steps 0 --scene_flow_weight ${flow_prior_weight} \
                    --spiral_radius 0.03 --warped_color_weight ${warped_color_weight} \
                    " $additional_args
    done
done
