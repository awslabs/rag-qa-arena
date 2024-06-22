from collections import Counter, defaultdict
import json
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
import re
import numpy as np
from argparse import ArgumentParser

response_1_regex = re.compile(r'<answer 1>\n.+\n</answer 1>', flags=re.DOTALL)
response_2_regex = re.compile(r'<answer 2>\n.+\n</answer 2>', flags=re.DOTALL)
rating_regex = re.compile(r'<rating>.*\d+.*</rating>', flags=re.DOTALL)

def cal_score(label, models):
    # better
    if label == models[0]:
        return 1
    # worse
    elif label == models[1]:
        return -1
    # tie
    else:
        return 0

def get_majority_vote(votes):
    # all votes agree --> strong agremment
    if len(set(votes)) == 1:
        return votes[0], 'strong'
    # no majority vote, consider as a tie
    elif len(set(votes)) == len(votes):
        return 0, ''
    # there is diagreement, but majority vote exists --> 'weak' agreemment
    else:
        counter = Counter(votes)
        return counter.most_common()[0][0], 'weak'
    
def aggregate_evaluators(target_models, eval_models, root="./", domain="", print_individual=True):
    
    agg_data = {}
    agg_counter = Counter()
    len_counter = defaultdict(list)

    for m, model in enumerate(eval_models):
        print(f">>>>>{'_'.join(target_models)}_eval_by_{model}<<<<<")
        counter = Counter()
        total, fail, identical, missing = 0, 0, 0, 0
        all_queries = []

        file_path = root
        if domain:
            file_path += f"{domain}_"

        with open(f"{file_path}{'_'.join(target_models)}_eval_by_{model}.json") as infile:
            for ex in json.load(infile):
                order = ex['order']
                r1 = re.findall(response_1_regex, ex['prompt'])
                r2 = re.findall(response_2_regex, ex['prompt'])
                if not r2 or not r1:
                    missing += 1
                    ex['pred'] == '<rating>1</rating>' if not r2 else '<rating>2</rating>'
                else:
                    r1 = r1[0].replace('<answer 1>\n', '').replace('\n</answer 1>', '')
                    r2 = r2[0].replace('<answer 2>\n', '').replace('\n</answer 2>', '')

                    if "I couldn't find an answer" not in r1 and "I couldn't find an answer" not in r2:
                        len_counter[order['1']].append(len(r1.split(' ')))
                        len_counter[order['2']].append(len(r2.split(' ')))
                
                if r1 == r2:
                    identical += 1

                total += 1
            
                score = 0
                if 'pred' not in ex or ex['pred'] == 'FAIL TO GENERATE ANS.' or '<rating>' not in ex['pred']:
                    fail += 1
                    counter['tie'] += 1
                try:
                    rating = re.findall(rating_regex, ex['pred'])[0]
                except:
                    rating = ex['pred']

                for i in range(0, 3):
                    if str(i) in rating:
                        score = i
                        if i == 0:
                            counter['tie'] += 1
                            agg_counter['tie'] += 1
                        else:
                            counter[order[str(i)].replace('/', '-')] += 1
                            agg_counter[order[str(i)].replace('/', '-')] += 1
                
                if m == 0:
                    agg_data[ex['query']] = {'id': ex.get('pair_id', ''),
                                             'query': ex['query'],
                                             'passages': ex.get('passages', []),
                                             'order': ex['order'],
                                             'response_1': r1,
                                             'response_2': r2,
                                             'scores': [score],
                                             'reference': ex.get('reference', '')}
                else:
                    assert ex['query'] == agg_data[ex['query']]['query']
                    if 'passages' in ex:
                        assert ex['passages'] == agg_data[ex['query']]['passages']
                    agg_data[ex['query']]['scores'].append(score)

                all_queries.append(ex['query'])

        if print_individual:
            print(f"Failed: {fail}; Missing: {missing}; Identical: {identical}; Total: {total}")
            for m in counter:
                if m != 'tie':
                    print(m, counter[m], round(counter[m]/total, 3), round(np.mean(len_counter[m]), 2))

    return agg_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--eval_model', default='gpt-4-0125-preview')
    parser.add_argument('--root', default='data/human_eval/')
    parser.add_argument('--human_eval_file', default='human_evaluations.json')
    args = parser.parse_args()
    
    targets = [['LFRQA',  'RQA'], ['RQA', 'gpt-4'], ['LFRQA', 'gpt-4'], ['LFRQA', 'mixtral-large'], ['LFRQA', 'llama-3']]

    for target in targets:

        predictions = aggregate_evaluators(target, [args.eval_model], root=args.root, print_individual=True)

        with open(f'{args.root}{args.human_eval_file}') as infile:
            annotations = json.load(infile)

        preds = {}
        for q, ex in predictions.items():          
            vote, _ = get_majority_vote(ex['scores'])
            
            if vote != 0:
                preds[ex['id']] = ex['order'][str(vote)]
            else:
                preds[ex['id']] = 'Tie'

        ann1, ann2 = [], []
        gold_counter = Counter()
        for i, p in preds.items():

            models = i.split('_')[-2:]
            g = cal_score(annotations[i]['majority'], models)
            gold_counter[annotations[i]['majority']] += 1
            p = cal_score(p, models)
            ann1.append(g)
            ann2.append(p)

        print('\nHuman Eval Results')
        for k, v in gold_counter.items():
            print(k, v, v/len(ann1))
        print("\nCohen's Kappa:", cohen_kappa_score(ann1, ann2))
        print("Pearson Correlation:", pearsonr(ann1, ann2))