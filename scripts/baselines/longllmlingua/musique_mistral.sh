

nohup python runner.py \
    --dataset_name musique \
    --train_data_path no_path \
    --test_data_path output/longllmlingua/longllmlingua_compress_results_musique_llama2.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/musique/output_longllmlingua_mistral.log 2>&1 &



nohup python runner.py \
    --dataset_name musique \
    --train_data_path no_path \
    --test_data_path output/longllmlingua/longllmlingua_compress_results_musique_llama2.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/musique/output_longllmlingua_llama2.log 2>&1 &