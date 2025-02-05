#!/bin/bash

python train_model.py -d test/training_data -m test/model -v
python run_model.py -d test/holdout_data -m test/model -o test/holdout_outputs -v
python evaluate_model.py -d test/holdout_data -o test/holdout_outputs

rm test/holdout_outputs/*
rm test/model/*
