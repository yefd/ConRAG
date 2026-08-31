


nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_small_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 1536 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/mrag/output_conrag_standard_mistral_small.log 2>&1 &

nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_large_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 3072 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/mrag/output_conrag_standard_mistral_large.log 2>&1 &




nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_large_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_rag_mistral_large.log 2>&1 ;

nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_small_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_rag_mistral_small.log 2>&1 ;