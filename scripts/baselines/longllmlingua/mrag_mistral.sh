

nohup python runner.py \
    --dataset_name mrag \
    --input_path output/longllmlingua/longllmlingua_compress_results_mrag_llama2.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_longllmlingua_mistral.log 2>&1 &



nohup python runner.py \
    --dataset_name mrag \
    --input_path output/longllmlingua/longllmlingua_compress_results_mrag_llama2.pkl \
    --model_name models/longchat-7b-v1.5-32k \
    --max_prompt_length 32000 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_longllmlingua_llama2_2.log 2>&1 &