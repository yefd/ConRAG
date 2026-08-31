


model_type="recomp"
date="20250101"

dataset_name="hotpotqa"

nohup python baseline_recomp_extractive.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --dataset_save_path dataset/hotpotqa/hotpot_train_v1.1_recomp_single.pkl \
    >  logs/$model_type/output_${date}_${dataset_name}_extractive_single.log 2>&1 ;




model_type="recomp"
date="20250101"

dataset_name="2wiki"

nohup python baseline_recomp_extractive.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --dataset_save_path dataset/2wikimultihop/2wikimultihop_dev_recomp_single.pkl \
    >  logs/$model_type/output_${date}_${dataset_name}_extractive_single.log 2>&1 ;
