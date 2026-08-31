


nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_bge_rerank_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 1024 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/discussion/diff_r/output_mrag_conrag_standard_mistral_bge_rerank_e1.log 2>&1 &

nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_bge_rerank_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 1024 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/discussion/diff_r/output_mrag_1208_conrag_standard_mistral_bge_rerank_e1.log 2>&1 &

nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_bert_nft_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/mrag/output_conrag_standard_mistral_bert_nft_e1.log 2>&1 ;

nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_contriever_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/mrag/output_conrag_standard_mistral_contriever_e1.log 2>&1 ;

nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_bge_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 1024 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/mrag/output_conrag_standard_mistral_bge_large_e1.log 2>&1 ;