


model_type="recomp"
date="20250101"
dataset_name="mrag"

nohup python baseline_recomp_extractive.py \
    --dataset_name $dataset_name \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --dataset_save_path dataset/multihop_rag/multihop_rag_recomp_single.pkl \
    >  logs/$model_type/output_${date}_${dataset_name}_extractive_single.log 2>&1 ;