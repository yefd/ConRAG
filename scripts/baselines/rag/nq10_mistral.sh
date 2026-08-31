

nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_10/output_rag_mistral.log 2>&1 &



nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_10/output_rag_llama2.log 2>&1 &
    
nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert.pkl \
    --model_name models/Llama-2-13b-chat-hf \
    --max_prompt_length 4096 \
    --hidden_size 5120 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type instruction \
    >  logs/nq_10/output_rag_llama_13b_instruction.log 2>&1 &

nohup python runner.py \
    --dataset_name nq_20 \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --input_dim 768 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_20/output_rag_llama2.log 2>&1 &

nohup python runner.py \
    --dataset_name nq_20 \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 8192 \
    --freeze_llm \
    --num_k 20 \
    --input_dim 768 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_20/output_rag_mistral.log 2>&1 &