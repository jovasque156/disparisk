#!/bin/bash

dataset="hate_speech"
split="train_val_test"
stratify_by="label"

echo "==============TRAIN FOR FAMILY: CLASS_LABEL==============="
label="class_label"
sensitive="dialect_class"
path_to_dataset="/path/to/data_sets/${dataset}/preprocessed/label_${label}_sensitive_${sensitive}/${split}/stratify_by_${stratify_by}/"

for model in bert-base-cased bert-large-cased microsoft/deberta-v3-base microsoft/deberta-v3-large facebook/bart-base facebook/bart-large gpt2 gpt2-large; do
    batch_size=32
    learning_rate=5e-5
    if [ "$model" = "facebook/bart-large" ] || [ "$model" = "microsoft/deberta-v3-large"  ] || [ "$model" = "gpt2" ] || [ "$model" = "bert-large-cased" ] || [ "$model" = "gpt2-large" ]; then
        batch_size=16
        learning_rate=5e-6
    fi

    echo "Running experiments with model_name: $model"
    echo ">starting std version"        
    path_to_save="/path/to/results/${dataset}/${model}/${label}_std"
    python3 train_on_text.py --path_to_dataset $path_to_dataset --path_to_save_results $path_to_save --batch_size $batch_size --model_name $model --pretrained --num_epochs 5
    
    echo ">computing pvis"
    for set in val test; do
        echo "> ${set}"
        python3 compute_basic_pvis_on_text.py --dataset_name $dataset --label $label --sensitive $sensitive --split train_val_test --stratify_by label --model_name $model --set $set --batch_size $batch_size
    done
    echo "Finished experiment"
    echo "------------------------------------------"
done