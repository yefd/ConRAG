


model_type="recomp"
date="20250101"

dataset_name="nq_10"
nohup python baseline_recomp_extractive.py \
    --dataset_name $dataset_name \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert.pkl \
    --dataset_save_path dataset/nq/nq-open-10_total_documents_recomp.pkl \
    >  logs/$model_type/output_${date}_${dataset_name}_extractive.log 2>&1 ;

dataset_name="nq_20"
nohup python baseline_recomp_extractive.py \
    --dataset_name $dataset_name \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert.pkl \
    --dataset_save_path dataset/nq/nq-open-20_total_documents_recomp.pkl \
    --num_k 10 \
    >  logs/$model_type/output_${date}_${dataset_name}_extractive_top10.log 2>&1 ;
