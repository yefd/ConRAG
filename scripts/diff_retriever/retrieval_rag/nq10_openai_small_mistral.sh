

nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_embedding_small_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_10/output_rag_mistral_openai_small.log 2>&1 &