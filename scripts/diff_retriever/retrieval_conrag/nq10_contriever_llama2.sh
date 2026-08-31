

nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_contriever_emb.pkl \
    --standard_prompt \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_nq_contriever \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --save_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/nq_10/output_conrag_llama2_contriever_e1.log 2>&1 &