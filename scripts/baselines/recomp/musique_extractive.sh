


model_type="recomp"
date="20250101"

dataset_name="musique"

nohup python baseline_recomp_extractive.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --dataset_save_path dataset/musique/musique_ans_v1.0_dev_recomp_single_top10.pkl \
    --num_k 10 \
    >  logs/$model_type/output_${date}_${dataset_name}_extractive_single_top10.log 2>&1 &
