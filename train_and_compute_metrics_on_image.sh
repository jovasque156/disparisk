dataset="facet"
split="train_val_test"
stratify_by="label"

echo "==============TRAIN FOR FAMILY: CLASS1==============="
label="class1"
sensitive="gender_presentation_masc"
path_to_img_df="/path/to/data_sets/${dataset}/preprocessed/label_${label}_sensitive_${sensitive}/${split}/stratify_by_${stratify_by}/"
path_to_train_dataset="/path/to/data_sets/${dataset}/"


for class in nurse; do
    echo "Running for: $class"
    for model in googlenet inception_v3 mobilenet_v3_small mobilenet_v2 resnet18 resnet152 ViT_32 ViT_16; do
        batch_size=32
        if [ "$model" = "ViT_16" ]; then
            batch_size=16
        fi
        
        echo "Running experiments with model_name: $model"
        echo ">starting std version"        
        path_to_save="/path/to/results/${dataset}/${model}/${label}_${class}_std"
        python3 train_on_image.py --path_to_train_dataset $path_to_train_dataset --path_to_img_df $path_to_img_df --path_to_save_results $path_to_save --batch_size $batch_size --model_name $model --pretrained --num_epochs 5 --binary --class_to_binarize $class
        
        echo ">starting PVI computation"
        batch_size=254
        if [ "$model" = "ViT_16" ]; then
            batch_size=54
        fi
        for set in test val; do
            echo "> on dataset $set"
            python3 compute_basic_pvis_on_image.py --dataset_name $dataset --label $label --sensitive $sensitive --split train_val_test --stratify_by label --model_name $model --set $set --batch_size $batch_size --pretrained --binary --class_to_binarize $class
        done
    done
done