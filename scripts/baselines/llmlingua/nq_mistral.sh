nohup python runner.py \
    --dataset_name nq_10 \
    --input_path output/longllmlingua/nq10_compress_result.json \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_10/output_llmlingua_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name nq_10 \
    --input_path output/longllmlingua/nq10_compress_result_llama2.json \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_10/output_llmlingua_llama2.log 2>&1 &

nohup python runner.py \
    --dataset_name nq_20 \
    --input_path output/longllmlingua/nq20_compress_result.json \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_20/output_llmlingua_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name nq_20 \
    --input_path output/longllmlingua/nq20_compress_result_llama2.json \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/nq_20/output_llmlingua_llama2.log 2>&1 &