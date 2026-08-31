

nohup python runner.py \
    --dataset_name lkqa \
    --input_path dataset/lkqa/lkqa_bge_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
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
    >  logs/lkqa/output_conrag_standard_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name lkqa \
    --input_path dataset/lkqa/lkqa_bge_emb.pkl \
    --standard_prompt \
    --model_name models/Qwen2-7B-Instruct \
    --max_prompt_length 4096 \
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
    >  logs/lkqa/output_conrag_standard_qwen.log 2>&1 ;