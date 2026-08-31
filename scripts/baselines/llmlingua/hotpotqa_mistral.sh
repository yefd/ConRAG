nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path no_path \
    --test_data_path output/longllmlingua/hotpotqa_compress_result.json \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/hotpotqa/output_llmlingua_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path no_path \
    --test_data_path output/longllmlingua/hotpotqa_compress_result_llama2.json \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/hotpotqa/output_llmlingua_llama2.log 2>&1 &