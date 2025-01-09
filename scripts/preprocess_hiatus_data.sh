# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash
# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash


if [ ! -d "/mnt/swordfish-pool2/nikhil/raw_hiatus" ]; then
    mkdir /mnt/swordfish-pool2/nikhil/raw_hiatus
fi

input_data_path="/mnt/swordfish-pool2/nikhil/hiatus/HRS_evaluation_samples/HRS1_english_long/TA2/HRS1_english_long_sample-0_crossGenre/data/"
groundtruth_path = "/mnt/swordfish-pool2/nikhil/hiatus/HRS_evaluation_samples/HRS1_english_long/TA2/HRS1_english_long_sample-0_crossGenre/groundtruth/"

python scripts/process_hiatus.py ${input_data_path} ${groundtruth_path}
