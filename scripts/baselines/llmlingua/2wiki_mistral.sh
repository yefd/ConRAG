nohup python runner.py \
    --dataset_name 2wiki \
    --train_data_path no_path \
    --test_data_path output/longllmlingua/2wiki_compress_result.json \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/2wiki/output_llmlingua_mistral_2wiki.log 2>&1 &


nohup python runner.py \
    --dataset_name 2wiki \
    --train_data_path no_path \
    --test_data_path output/longllmlingua/2wiki_compress_result_llama2.json \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/2wiki/output_llmlingua_llama2.log 2>&1 &