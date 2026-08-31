


model_type="recomp"
date="20250101"

dataset_name="2wiki"

nohup python baseline_recomp_extractive.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --dataset_save_path dataset/2wikimultihop/2wikimultihop_dev_recomp.pkl \
    >  logs/$model_type/output_${date}_${dataset_name}_extractive.log 2>&1 ;
