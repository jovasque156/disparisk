#!/bin/bash
dataset="census_income_kdd"
split="train_val_test"
stratify_by="label"
scaler="standard"

echo "==============TRAIN FOR FAMILY: CLASS==============="
label="class"
sensitive="sex"
path_to_ds="/path/to/data_sets/${dataset}/raw/preprocessed/label_${label}_sensitive_${sensitive}/train_val_test/stratify_by_${stratify_by}/${scaler}/"

for class in nurse; do
    echo "Running for: $class"
    for model in h1 h4 h1_relu h4_relu h1_sigmoid h4_sigmoid h1_gelu h4_gelu; do
        echo "Running experiments with model_config: $model"
        echo ">starting std version"        
        path_to_save="/path/to/results/${dataset}/${model}/${label}_std"
        python3 train_on_tabular.py --path_to_train_dataset $path_to_ds --path_to_save_results $path_to_save --batch_size 32 --model_config $model --num_epochs 5
        
        echo ">starting PVI computation"
        for set in test val; do
            echo "> on dataset $set"
            python3 compute_basic_pvis_on_tabular.py --dataset_name $dataset --label  $label --sensitive $sensitive --split train_val_test --stratify_by label --scaler standard --model_config $model --set $set
        done
    done
done