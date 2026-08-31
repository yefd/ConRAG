nohup python runner.py \
    --dataset_name mrag \
    --input_path output/longllmlingua/mrag_compress_result.json \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_llmlingua_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name mrag \
    --input_path output/longllmlingua/mrag_compress_result_llama2.json \
    --model_name models/longchat-7b-v1.5-32k \
    --max_prompt_length 32000 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_llmlingua_llama2.log 2>&1 &