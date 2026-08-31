model_type="adaptive_rag"
date="20250101"

dataset_name="hotpotqa"

nohup python baseline_adaptive_rag.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --classifier_name models/flan-t5-base \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_rag_mistral.log 2>&1 ;

nohup python baseline_adaptive_rag.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --classifier_name models/flan-t5-base \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_rag_llama2.log 2>&1 ;