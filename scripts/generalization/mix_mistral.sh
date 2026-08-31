


nohup python runner.py \
    --dataset_names nq_10 musique 2wiki hotpotqa \
    --paths \
    dataset/nq/nq-open-10_total_documents_gold_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_train_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_dev_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_train_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_dev_contriever_emb.pkl \
    dataset/hotpotqa/hotpot_train_v1.1_contriever_8k_emb.pkl \
    dataset/hotpotqa/hotpot_dev_distractor_v1_contriever_2k_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_gen \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --save_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/discussion/output_conrag_generalization_mistral.log 2>&1 &





nohup python runner.py \
    --dataset_names nq_10 musique 2wiki hotpotqa \
    --paths \
    dataset/nq/nq-open-10_total_documents_gold_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_train_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_dev_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_train_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_dev_contriever_emb.pkl \
    dataset/hotpotqa/hotpot_train_v1.1_contriever_8k_emb.pkl \
    dataset/hotpotqa/hotpot_dev_distractor_v1_contriever_2k_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_gen \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/discussion/generalization/output_conrag_generalization_mistral_e5_00.log 2>&1 &

nohup python runner.py \
    --dataset_names nq_10 musique 2wiki hotpotqa \
    --paths \
    dataset/nq/nq-open-10_total_documents_gold_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_train_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_dev_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_train_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_dev_contriever_emb.pkl \
    dataset/hotpotqa/hotpot_train_v1.1_contriever_8k_emb.pkl \
    dataset/hotpotqa/hotpot_dev_distractor_v1_contriever_2k_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_gen \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/discussion/generalization/output_conrag_generalization_mistral_e5_11.log 2>&1 &

nohup python runner.py \
    --dataset_names nq_10 musique 2wiki hotpotqa \
    --paths \
    dataset/nq/nq-open-10_total_documents_gold_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_train_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_dev_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_train_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_dev_contriever_emb.pkl \
    dataset/hotpotqa/hotpot_train_v1.1_contriever_8k_emb.pkl \
    dataset/hotpotqa/hotpot_dev_distractor_v1_contriever_2k_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_gen \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/discussion/generalization/output_conrag_generalization_mistral_2.log 2>&1 &

nohup python runner.py \
    --dataset_names nq_10 musique 2wiki hotpotqa \
    --paths \
    dataset/nq/nq-open-10_total_documents_gold_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_train_contriever_emb.pkl \
    dataset/musique/musique_ans_v1.0_dev_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_train_contriever_emb.pkl \
    dataset/2wikimultihop/2wikimultihop_dev_contriever_emb.pkl \
    dataset/hotpotqa/hotpot_train_v1.1_contriever_8k_emb.pkl \
    dataset/hotpotqa/hotpot_dev_distractor_v1_contriever_2k_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_gen \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/discussion/generalization/output_conrag_generalization_mistral_2.log 2>&1 &