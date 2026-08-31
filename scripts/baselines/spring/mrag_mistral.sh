


# sleep 7200;

nohup python baseline_spring.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --num_k 10 \
    --input_dim 768 \
    --model_type spring \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/spring/output_mrag_mistral.log 2>&1 ;

nohup python baseline_spring.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/longchat-7b-v1.5-32k \
    --max_prompt_length 32000 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --num_k 10 \
    --input_dim 768 \
    --model_type spring \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/spring/output_mrag_llama2.log 2>&1 ;