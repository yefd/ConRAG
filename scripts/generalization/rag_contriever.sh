


nohup python runner.py \
    --dataset_name musique \
    --train_data_path dataset/musique/musique_ans_v1.0_train_contriever_emb.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_contriever_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/discussion/generalization/output_musique_rag_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_contriever_8k_emb.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_contriever_2k_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/discussion/generalization/output_hotpotqa_rag_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_contriever_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/discussion/generalization/output_nq10_rag_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_contriever_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_contriever_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/discussion/generalization/output_2wiki_rag_mistral.log 2>&1 &