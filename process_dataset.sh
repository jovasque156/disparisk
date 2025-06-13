echo "=====Processing KDD to train, val, test===="
echo "label: class, sensitive: sex, random_split, standard"
python3 dataset_preprocessing.py --dataset_name census_income_kdd --label class --sensitive sex --random_split --val_size 0.1 --test_size 0.2 --stratify_by label --scaler standard
echo "---"

echo ""

echo "=====Processing FACET to train, val, test===="
dataset_name="facet"
label="class1"
sensitive="gender_presentation_masc"
echo "label: ${label}, sensitive: ${sensitive}, random_split"
python3 dataset_preprocessing.py --dataset_name $dataset_name --label $label --sensitive $sensitive --random_split --val_size 0.1 --test_size 0.2 --stratify_by label
echo "---"

echo ""

echo "=====Processing HATE SPEECH to train, val, test===="
dataset_name="hate_speech"
label="class_label"
sensitive="dialect_class"
echo "label: ${label}, sensitive: ${sensitive}, random_split"
python3 dataset_preprocessing.py --dataset_name $dataset_name --label $label --sensitive $sensitive --random_split --val_size 0.1 --test_size 0.2 --stratify_by label
echo "---"