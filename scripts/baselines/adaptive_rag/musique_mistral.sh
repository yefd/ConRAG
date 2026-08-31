model_type="adaptive_rag"
date="20250101"

dataset_name="musique"

nohup python baseline_adaptive_rag.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --classifier_name models/flan-t5-base \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral.log 2>&1 ;

nohup python baseline_adaptive_rag.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --classifier_name models/flan-t5-base \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_llama2.log 2>&1 ;