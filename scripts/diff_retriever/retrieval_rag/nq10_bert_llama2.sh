

nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert_notrain.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_10/output_rag_bert_llama2.log 2>&1 &