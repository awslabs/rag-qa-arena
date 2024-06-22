from compute_correlation import aggregate_evaluators, get_majority_vote
from collections import Counter
from itertools import combinations
import json
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--eval_model', default='gpt-4-0125-preview')
    parser.add_argument('--root', default='eval_results/lfrqa/5_psgs/')
    parser.add_argument('--output_dir', default='eval_results/lfrqa/')
    parser.add_argument('--use_complete_pairs', action='store_true')
    parser.add_argument('--sub_dirs', nargs='+', default='', help='in case we save eval results in sub-directories')
    args = parser.parse_args()
    
    target_models = [
                 'gpt-4o',
                 'gpt-4-turbo',
                 'gpt-4-0125-preview',
                 'mistralai-Mixtral-8x22B-Instruct-v0.1',
                 'mistralai-Mixtral-8x7B-Instruct-v0.1',
                 'meta-llama-Meta-Llama-3-70B-Instruct',
                 'meta-llama-Meta-Llama-3-8B-Instruct',
                 'CohereForAI-c4ai-command-r-plus',
                 'CohereForAI-c4ai-command-r-v01',
                 'Qwen-Qwen1.5-110B-Chat',
                 'Qwen-Qwen1.5-32B-Chat'
                 ]
    
    # battles data for Elo rating
    battles = {'model_a': [], 'model_b': [], 'winner': []}

    if args.use_complete_pairs:
        target_models += [list(pair) for pair in combinations(target_models, 2)]

    for target_model in target_models:
        target_model = [target_model] if isinstance(target_model, str) else target_model

        if len(target_model) == 1:
            filedirs =  [args.root]
        else:
            filedirs = [f"{args.root}{subdir}/" for subdir in args.sub_dirs]
            
        for filedir in filedirs:
            all_counter, all_total = Counter(), 0
            for domain in ['bioasq', 'fiqa', 'lifestyle', 'recreation', 'technology', 'science', 'writing']:
                print(f"Eval {domain}...")
                try:
                    predictions = aggregate_evaluators(target_model, [args.eval_model], root=filedir, domain=domain, print_individual=False)
                except:
                    print(f">>Fail aggregating results!")
                    continue

                counter = Counter()
                total = 0

                for q, ex in predictions.items():          
                    
                    vote, signal = get_majority_vote(ex['scores'])
                    if len(target_model) == 1:
                        if ex['order']['1'] == 'LFRQA':
                            battles['model_a'].append('LFRQA')
                        else:
                            battles['model_a'].append(target_model[0])

                        if ex['order']['2'] == 'LFRQA':
                            battles['model_b'].append('LFRQA')
                        else:
                            battles['model_b'].append(target_model[0])
                    else:
                        battles['model_a'].append(ex['order']["1"].replace('/', '-'))
                        battles['model_b'].append(ex['order']["2"].replace('/', '-'))
                    
                    if vote != 0:
                        counter[ex['order'][str(vote)].replace('/', '-')] += 1
                        winner = 'model_a' if vote == 1 else 'model_b'
                    else:
                        counter['tie'] += 1
                        winner = 'tie'

                    battles['winner'].append(winner)
                    total += 1
                    all_total += 1

                print(total)
                for m, v in counter.items():
                    all_counter[m] += v
                    if m != 'LFRQA' and m != 'tie':
                        print(m, counter[m], round(counter[m]/total, 3))
                        no_worse = counter[m] + counter['tie']
                        print('win+tie', no_worse, round(no_worse/total, 3))
            
            print("=" * 20)
            print(all_total)
            for m, v in all_counter.items():
                if m != 'LFRQA' and m != 'tie':
                    print(m, all_counter[m], round(all_counter[m]/all_total, 3))
                    no_worse = all_counter[m] + all_counter['tie']
                    print('win+tie', no_worse, round(no_worse/all_total, 3))
            print("=" * 50)

    with open(f"{args.output_dir}/all_battles.json", 'w') as outfile:
        json.dump(battles, outfile, indent=2)
