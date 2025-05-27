import sys
import os
import numpy as np
import json
from collections import defaultdict
from probe import PROBE

# This function computes the probability of a triplet being true based on the MLN outputs.
def mln_triplet_prob(h, r, t, hrt2p):
    # KGE algorithms tend to predict triplets like (e, r, e), which are less likely in practice.
    # Therefore, we give a penalty to such triplets, which yields some improvement.
    if h == t:
        if hrt2p.get((h, r, t), 0) < 0.5:
            return -100
        return hrt2p[(h, r, t)]
    else:
        if (h, r, t) in hrt2p:
            return hrt2p[(h, r, t)]
        return 0.5

# This function reads the outputs from MLN and KGE to do evaluation.
# Here, the parameter weight controls the relative weights of both models.
def evaluate(mln_pred_file, kge_pred_file, output_file, weight, count_info_dict_trn, entity2id, logger1, args, _mode, _k):
    with open(output_file, 'w') as fo:
        fo.write('starting evaluation...\n')
    
    hit1 = 0
    hit3 = 0
    hit10 = 0
    mr = 0
    mrr = 0
    cn = 0

    just_raw_ranks = []
    
    rank_per_class = defaultdict(list)

    hrt2p = dict()
    with open(mln_pred_file, 'r') as fi:
        for line in fi:
            h, r, t, p = line.strip().split('\t')[0:4]
            hrt2p[(h, r, t)] = float(p)

    with open(kge_pred_file, 'r') as fi:
        target = None
        while True:
            # truth and preds contents can be found in pred_kge_valid file
            truth = fi.readline()
            preds = fi.readline()

            if (not truth) or (not preds):
                break

            truth = truth.strip().split()
            preds = preds.strip().split()

            h, r, t, mode, original_ranking = truth[0:5]
            original_ranking = int(original_ranking)
            if mode == 'h':
                target = entity2id[h]
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(e, r, t, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                pred_scores = np.array(list(preds[i][1] for i in range(len(preds))))
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == h:
                        just_raw_ranks.append(k + 1)

                        target_score = pred_scores[k]
                        higher_than_target = (pred_scores > target_score).sum()
                        ties_mean = ((pred_scores == target_score).sum() + 1) / 2
                        ranking = higher_than_target + ties_mean 
                        break
                if ranking == -1:
                    ranking = original_ranking
                    just_raw_ranks.append(ranking)

            if mode == 't':
                target = entity2id[t]
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(h, r, e, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                pred_scores = np.array(list(preds[i][1] for i in range(len(preds))))
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == h:
                        just_raw_ranks.append(k + 1)

                        target_score = pred_scores[k]
                        higher_than_target = (pred_scores > target_score).sum()
                        ties_mean = ((pred_scores == target_score).sum() + 1) / 2
                        ranking = higher_than_target + ties_mean
                        break
                if ranking == -1:
                    ranking = original_ranking
                    just_raw_ranks.append(ranking)
            
            rank_per_class[target].append(ranking)
            if ranking <= 1:
                hit1 += 1
            if ranking <=3:
                hit3 += 1
            if ranking <= 10:
                hit10 += 1
            mr += ranking
            mrr += 1.0 / ranking
            cn += 1

    count_info_dict_tst = {k: len(v) for k, v in rank_per_class.items()}
    met = PROBE(rank_per_class, count_info_dict_trn, count_info_dict_tst, len(entity2id))
    met.set_raw_ranks(rank_per_class)
    alphas, betas = [0.25, 0.5, 1.0, 2.0], [0.8, 0.4, 0.2, 0.0]
    for a in alphas:
        for b in betas:
            logger1.info(f'PROBE(alpha={a},beta={b}) : {met.calculate_final_metric(a, b)}')

    logger1.info(f'MRR without breaking ties : {sum(1/i for i in just_raw_ranks) / len(just_raw_ranks)}')

    mr /= cn
    mrr /= cn
    hit1 /= cn
    hit3 /= cn
    hit10 /= cn

    print('MR: ', mr)
    print('MRR: ', mrr)
    print('Hit@1: ', hit1)
    print('Hit@3: ', hit3)
    print('Hit@10: ', hit10)

    total_metric = {}
    total_metric['MR'] = mr
    total_metric['MRR'] = mrr
    total_metric['HITS@1'] = hit1
    total_metric['HITS@3'] = hit3
    total_metric['HITS@10'] = hit10
    with open(output_file, 'a') as fo:
        fo.write("Total metrics : {}\n".format(str(total_metric)))

    with open(output_file, 'a') as fo:
        fo.write('MR: {}\n'.format(mr))
        fo.write('MRR: {}\n'.format(mrr))
        fo.write('Hit@1: {}\n'.format(hit1))
        fo.write('Hit@3: {}\n'.format(hit3))
        fo.write('Hit@10: {}\n'.format(hit10))
    logger1.info(f'MR : {mr}')
    logger1.info(f'MRR : {mrr}')
    logger1.info(f'Hits@1 : {hit1}')
    logger1.info(f'Hits@3 : {hit3}')
    logger1.info(f'Hits@10 : {hit10}')
   

def augment_triplet(pred_file, trip_file, out_file, threshold):
    with open(pred_file, 'r') as fi:
        data = []
        for line in fi:
            l = line.strip().split()
            data += [(l[0], l[1], l[2], float(l[3]))]

    with open(trip_file, 'r') as fi:
        trip = set()
        for line in fi:
            l = line.strip().split()
            trip.add((l[0], l[1], l[2]))

        for tp in data:
            if tp[3] < threshold:
                continue
            trip.add((tp[0], tp[1], tp[2]))

    with open(out_file, 'w') as fo:
        for h, r, t in trip:
            fo.write('{}\t{}\t{}\n'.format(h, r, t))
