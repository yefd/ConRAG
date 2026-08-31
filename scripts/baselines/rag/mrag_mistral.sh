

nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_rag_mistral_flash.log 2>&1 &



nohup python runner.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/longchat-7b-v1.5-32k \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/mrag/output_rag_longchat.log 2>&1 &