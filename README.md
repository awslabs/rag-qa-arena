# RAG-QA Arena

## 0. Preliminary

### 0.1 Install Environment
```
conda create -n ragqa-arena python=3.10.0
pip install -r requirements.txt
```

### 0.2 Download Data
Data can be downloaded here: https://drive.google.com/drive/folders/1qhKEbGgBfrPzqGvhnzVoTvi3a6tiMyDP?usp=sharing. Save them into the `data/` folder.

Query with citation can be downloaded in `data/annotations_{domain}_with_citation.jsonl`. The in-text citations are included in the field `"faithful_answer_w_citation"`, in the form of `[1, 2, 3]`, which indicate the corresponding index of the annotated gold documents, which are shown in the format of `doc_id` that can be mapped to the `doc_id` corresponding to [RobustQA document source](https://github.com/awslabs/robustqa-acl23).
Citations are also availble in the form of list of list in `"citation_numbers"`, which are listed in the same order of in-text citations.

There is a minor amount difference in `data/annotations_{domain}_with_citation.jsonl` with [previous version](https://drive.google.com/drive/folders/1qhKEbGgBfrPzqGvhnzVoTvi3a6tiMyDP?usp=sharing) that does not have citation annotation (statistics shown below). Please proceed with the version that satisfy your use case.

| **Domain** | **Original All Samples** | **With Citation** |
|------------|--------------------------|----------------------|
| Technology |                     2115 |                 2064 |
| Lifestyle  |                     2212 |                 2198 |
| Recreation |                     2096 |                 2090 |
| Writing    |                     2696 |                 2694 |
| Science    |                     1426 |                 1404 |
| FIQA       |                     3669 |                 3570 |

We only provide new annotations for this project. For underlying corpus, you need to 
1. Follow the instruction from RobustQA here: https://github.com/awslabs/robustqa-acl23?tab=readme-ov-file#raw-data--annotations to obtain raw documents and processed passages.
2. Follow instruction to run ColBERTv2 and retrieve passages: https://github.com/awslabs/robustqa-acl23?tab=readme-ov-file#colbertv2.
3. We provide 1 example per domain for final data format under `data/ans_gen/`.


### WARNING: to run the following scripts, you will need your own OpenAI access_token or have access to up to 8 A100 GPUs.

## 1. Answer Generation

### 1.1 GPT-4 Models
Modify the script by adding your `openai_key`.
```
bash generate_response_gpt.sh
```

### 1.2 Huggingface Models
Note that you will need up to 8 A100 GPUs, change models and set `clm_new_token_trigger` accordlingly in this script in order to parse output properly.
```
bash generate_response_hf.sh
```

Output files are saved under `data/pairwise_eval/`, which will be used as inputs to the next section `Pairwise Evaluation`.

## 2. Pairwise Evaluation
We select `gpt-4-0125-preview` as our final evaluator.

### 2.1 Compare with LFRQA directly.
Modify the script by adding your `openai_key`.
```
bash run_pairwise_eval_lfrqa.sh
```

### 2.2 Compare a pair of LLM generations.
Modify the script by adding your `openai_key`.
```
bash run_pairwise_eval_llms.sh
```

## 3. Analysis

After running Section 2, you can run the following script to see RAG-QA Arena results.

### 3.1 Main Results
```
python code/report_results.py
```
This script reports win and win+tie rate against LFRQA only.

### 3.2 Complete Pairs
```
python code/report_results.py --use_complete_pairs
```
This script reports win and win+tie rate for all comparison, and output an `all_battles.json` file that can be used in this Google Colab: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH to compute Elo Rating.

### 3.3 Replicating Existing Results
We've evaluated 11 leading models and their complete pairs for 1400 queries. Results can be downloaded from https://drive.google.com/drive/folders/1fnJ_PUd33go0BXq0ShY9ofkVYERpyzFu?usp=sharing. Run the following scripts to see results. Make sure results are in `./eval_results/` sub-folder.

1. Top 5 retrieved passages for 11 LLMs compared to all LFRQA answers.
```
python code/report_results.py \
        --root eval_results/lfrqa/5_psgs/
```

2. Top 5 retrieved passages for 11 LLMs compared to all LFRQA answers + 1400 queries with complete LLM pairwise comparisons.
```
python code/report_results.py
        --root eval_results/lfrqa/5_psgs/
        --use_complete_pairs 
        --sub_dirs complete_pair_batch_1 complete_pair_batch_2
```

3. Top 10 retrieved passages for 11 LLMs compared to all LFRQA answers.
```
python code/report_results.py
        --root eval_results/lfrqa/10_psgs/
```


### 3.4 Correlation between Human and Model-based Evaluation

In `data/human_eval`, we share two types of evaluation results,
- `human_evaluations.json` includes human evaluations results. There are three judgments in "raw" and we take the majority vote as the final vote.
- `{LFRQA|RQA}_{RQA|gpt-4|llama-3|mixtral-large}-eval_by_gpt4-0125-preview.json` is the LLM-based evaluators' results. In these files, you can find RobustQA and LFRQA's annotation, together with LLMs generated answers. We sampled 100 queries per domain, so 700 queries in total. For brevity, we do not repeat the specific script to generate model-based evaluation results for these queries. All details can be found in Section 2 above.

To check results, simply run `python code/compute_correlation.py`.


## Citation
```
@article{han2024ragqaarena,
  title={RAG-QA Arena: Evaluating Domain Robustness for Long-form Retrieval Augmented Question Answering},
  author={Rujun Han and Yuhao Zhang and Peng Qi and Yumo Xu and Jenyuan Wang and Lan Liu and William Yang Wang and Bonan Min and Vittorio Castelli},
  year={2024},
  journal={arXiv preprint arXiv:2407.13998},
  url={https://arxiv.org/abs/2407.13998}
}
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

