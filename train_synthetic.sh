#!/bin/bash

datadir="data/synthetic_scenes"
basedir="logs"


#  Define list for arguments per scene
scan_list=(
    "sliding_cube"
    "occlusion"
    "speed_test_texture"
    "speed_test_chair"
    "arcing_cube"
    "z_motion_speed_test"
    "acute_z_speed_test"
)

min_depth_fac_list=(
    "0.07"
    "0.03"
    "0.08"
    "0.08"
    "0.03"
    "0.06"
    "0.01"
)

max_depth_fac_list=(
    "0.24"
    "0.21"
    "0.32"
    "0.32"
    "0.38"
    "0.34"
    "0.52"
)


# Check if the lengths of the lists are the same
if [ ${#scan_list[@]} -ne ${#min_depth_fac_list[@]} ] || [ ${#scan_list[@]} -ne ${#max_depth_fac_list[@]} ]; then
    echo "Error: The lengths of scan_list, min_depth_fac_list, and max_depth_fac_list are not the same."
    exit 1
fi

target_scene=$1
additional_args=$2

# Loop through lists and execute the script
for ((i=0; i<${#scan_list[@]}; i++)); do
    current_scene=${scan_list[i]}
    # Skip if the current scene does not match the target scene
    if [ -n "$target_scene" ] && [ "$current_scene" != "$target_scene" ]; then
        continue
    fi
    echo "Processing scene: $current_scene"

    scan=${scan_list[i]}
    min_depth_fac=${min_depth_fac_list[i]}
    max_depth_fac=${max_depth_fac_list[i]}

    ./scripts/run_synthetic.sh 60 $datadir $basedir $scan $min_depth_fac $max_depth_fac $additional_args

    echo "Script with arguments: $datadir $basedir $scan $expname $min_depth_fac $max_depth_fac $additional_args done!"
done

echo "All jobs launched"