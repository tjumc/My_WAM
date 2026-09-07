#!/usr/bin/env bash

# Dataset directories present in the current Astribot LeRobot v3 inventory
# but absent from the supplied valid classification.

declare -ag ASTRIBOT_INVALID_TASKS=(
  Task01
  Task02
  bake_bread
  dishwasher
  hot_milk
  oven
  oven_bread_2
  oven_fx
  oven_new
  oven_serve_food
  random_folding_shirts
  wipe_the_table
)

declare -Ag ASTRIBOT_INVALID_DATASET_CATALOG=()
ASTRIBOT_INVALID_DATASET_CATALOG[Task01]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/Task01_fx_20260701_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/Task01_fx_20260702_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[Task02]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/Task02_fx_20260702_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/Task02_fx_20260703_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[bake_bread]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/bake_bread_fx_20260205_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/bake_bread_fx_20260206_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/bake_bread_fx_20260209_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/bake_bread_fx_20260210_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/bake_bread_fx_20260225_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[dishwasher]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_fx_20260312_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_fx_20260313_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_fx_20260316_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_fx_20260324_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_fx_20260327_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_fx_20260331_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_zjm_20260302_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_zjm_20260303_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_zjm_20260304_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_zjm_20260305_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[hot_milk]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/hot_milk_zjm_20260225_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/hot_milk_zjm_20260226_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/hot_milk_zjm_20260227_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/hot_milk_zjm_20260228_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[oven]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_zjm_20260316_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[oven_bread_2]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_bread_2_fx_20260519_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_bread_2_fx_20260520_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_bread_2_fx_20260521_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[oven_fx]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_fx_zjm_20260316_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_fx_zjm_20260317_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_fx_zjm_20260318_1_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_fx_zjm_20260318_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_fx_zjm_20260319_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_fx_zjm_20260320_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[oven_new]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_bxx_20260423_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_bxx_20260424_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_bxx_20260428_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_bxx_20260429_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_bxx_20260430_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_fx_20260414_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_fx_20260415_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_fx_20260416_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_fx_20260417_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_fx_20260421_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_new_fx_20260422_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[oven_serve_food]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_serve_food_fx_20260622_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/oven_serve_food_fx_20260623_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[random_folding_shirts]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/random_folding_shirts_zby_20260122_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/random_folding_shirts_zby_20260123_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/random_folding_shirts_zby_20260126_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/random_folding_shirts_zby_20260127_compressed'
ASTRIBOT_INVALID_DATASET_CATALOG[wipe_the_table]=$'/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/wipe_the_table_fx_20260625_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/wipe_the_table_fx_20260626_compressed\n/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/wipe_the_table_fx_20260629_compressed'

# Entries declared valid but not present in the current dataset inventory.
declare -ag ASTRIBOT_VALID_BUT_MISSING_DIRS=(
  /data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/fridge_chicken_fx_20260507_compressed_1
)

astribot_list_invalid_datasets() {
  local task_name
  local -a task_dirs
  for task_name in "${ASTRIBOT_INVALID_TASKS[@]}"; do
    mapfile -t task_dirs <<< "${ASTRIBOT_INVALID_DATASET_CATALOG[${task_name}]}"
    printf '%-28s %3d\n' "${task_name}" "${#task_dirs[@]}"
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Invalid task groups: %d, dataset directories: %d\n\n' \
    "${#ASTRIBOT_INVALID_TASKS[@]}" \
    "$(printf '%s\n' "${ASTRIBOT_INVALID_DATASET_CATALOG[@]}" | awk -F '\\n' '{n += NF} END {print n}')"
  printf '%-28s %s\n' TASK DIRECTORIES
  astribot_list_invalid_datasets
  printf '\nValid entries missing from inventory: %d\n' "${#ASTRIBOT_VALID_BUT_MISSING_DIRS[@]}"
  printf '  %s\n' "${ASTRIBOT_VALID_BUT_MISSING_DIRS[@]}"
fi

