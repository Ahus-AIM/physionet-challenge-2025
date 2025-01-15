#!/bin/bash

# Set the base folder
BASE_FOLDER="/data/physionet-challenge-2025/code15"

# Set the shared arguments
DATA_FILE="$BASE_FOLDER/exams.csv"
LABELS_FILE="$BASE_FOLDER/code15_chagas_labels/code15_chagas_labels.csv"
OUTPUT_FOLDER="$BASE_FOLDER/code15_wfdb"

# Loop through file indices 0 to 17
for i in {0..17}; do
    INPUT_FILES="$BASE_FOLDER/exams_part${i}.hdf5"
    echo "Processing file: $INPUT_FILES"
    python prepare_code15_data.py -i $INPUT_FILES -d $DATA_FILE -l $LABELS_FILE -o $OUTPUT_FOLDER
done

echo "All files processed."
