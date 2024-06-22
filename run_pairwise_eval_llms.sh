##### This script compares a pair of LLMs' outputs ######

model1=gpt-4o

openai_key=''
eval_model=gpt-4-0125-preview

n_passages=5
domains=(bioasq fiqa recreation technology science writing lifestyle)

for model2 in gpt-4-turbo mistralai/Mixtral-8x22B-Instruct-v0.1
do
    for i in "${!domains[@]}"
    do
        echo evaluating ${model1} and ${model2} using ${eval_model} for ${domains[i]}
        python code/evaluate_pair_responses.py \
            --eval_dir "data/pairwise_eval" \
            --model ${eval_model} \
            --eval_model1 ${model1} \
            --eval_model2 ${model2} \
            --model1_pred_file ${model1}/${domains[i]}_from_colbert_${n_passages}_psgs  \
            --model2_pred_file ${model2}/${domains[i]}_from_colbert_${n_passages}_psgs \
            --reference_file references \
            --template_config pairwise_lfrqa.cfg \
            --domain ${domains[i]} \
            --temperature 0.0 \
            --eval_input_save_dir eval_inputs/from_colbert \
            --api_key ${openai_key}
    done
done