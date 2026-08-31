


nohup python runner.py \
    --dataset_name nq_20 \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 20 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/nq_20/output_conrag_standard_llama2_e2.log 2>&1

nohup python runner.py \
    --dataset_name nq_20 \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 20 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/nq_20/output_conrag_standard_llama2_e1.log 2>&1