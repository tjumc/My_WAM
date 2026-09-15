#!/usr/bin/env bash

DATASET_ROOTPATH="/efs/share/1919650160032350208/projects/foundation_model/foundation_model_/datasets/astribot_lerobotv3"

declare -ag ASTRIBOT_DATASET_TASKS=(
    fridge
    dishwasher
    oven_put
    wash_clothes
)

declare -Ag ASTRIBOT_DATASET_CATALOG=()

ASTRIBOT_DATASET_CATALOG[fridge]="$DATASET_ROOTPATH/fridge_fruits_fx_20260509_compressed"

ASTRIBOT_DATASET_CATALOG[dishwasher]="$DATASET_ROOTPATH/dishwasher_2_fx_20260605_compressed"

ASTRIBOT_DATASET_CATALOG[oven_put]="$DATASET_ROOTPATH/oven_new_2_fx_20260518_compressed"

ASTRIBOT_DATASET_CATALOG[wash_clothes]="$DATASET_ROOTPATH/wash_clothes_bxx_20260506_compressed"