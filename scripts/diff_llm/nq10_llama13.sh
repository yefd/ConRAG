nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Llama-2-13b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --hidden_size 5120 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/nq_10/output_conrag_standard_llama_13b_e2.log 2>&1 &