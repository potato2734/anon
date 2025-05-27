import sys
import os
import datetime
from utils import augment_triplet, evaluate
import argparse
import logging
import torch
import numpy as np
import random
from datetime import datetime

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_logger(args):
    log_file = os.path.join('./log/', f'pLogicNet_{args.dataset}_{args.seed}.log')
    
    os.makedirs('./log/', exist_ok=True)

    logger = logging.getLogger(__name__)  # Use module-level logger
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
    
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--kge_model', type=str)
    parser.add_argument('--threshold_of_rule', type=float)
    parser.add_argument('--mln_threshold_of_triplet', type=float)
    parser.add_argument('--weight', type=float)
    parser.add_argument('-seed', type=int)
    return parser.parse_args(args)

args = parse_args()
set_random_seeds(random_seed=args.seed)
logger1 = set_logger(args)
logger1.info(f'seed={args.seed}')
dataset = f'data/{args.dataset}'
path = f'./record-{args.seed}'

gpu = args.gpu
print(gpu)
iterations = args.iterations

kge_model = args.kge_model
kge_batch = 1024
kge_neg = 256
kge_dim = 128
kge_gamma = 24
kge_alpha = 1
kge_lr = 0.001
kge_iters = 10000
kge_tbatch = 16
kge_reg = 0.0
kge_topk = 100

if kge_model == 'RotatE':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 128, 24.0, 1.0, 0.0001, 150000, 16
    if dataset.split('/')[-1] == 'FB15k237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 128, 6.0, 1.0, 0.00005, 100000, 16
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 128, 12.0, 0.5, 0.0001, 80000, 8
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 128, 3.0, 1.0, 0.00005, 80000, 8

mln_threshold_of_rule = args.threshold_of_rule
mln_threshold_of_triplet = args.mln_threshold_of_triplet
weight = args.weight

mln_iters = 1000
mln_lr = 0.0001
mln_threads = 5

with open(os.path.join(dataset, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid) 
nentity = len(entity2id)

count_info_dict = {}
eid2count = {}

for i in range(nentity):
    eid2count[i] = 0
with open(dataset+'/train.txt') as fin:
    for line in fin:
        h, r, t = line.strip().split('\t')
        eid2count[entity2id[h]] += 1
        eid2count[entity2id[t]] += 1
count_info_dict = dict(sorted(eid2count.items(), key=lambda item : -item[1]))


# ------------------------------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def cmd_kge(workspace_path, model):
    if model == 'RotatE':
        return 'bash ./kge/kge.sh train {} {} {} {} {} {} {} {} {} {} {} {} {} -de'.format(model, dataset, gpu, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk)
    if model == 'TransE':
        return 'bash ./kge/kge.sh train {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(model, dataset, gpu, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk)
    if model == 'DistMult':
        return 'bash ./kge/kge.sh train {} {} {} {} {} {} {} {} {} {} {} {} {} -r {}'.format(model, dataset, gpu, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk, kge_reg)
    if model == 'ComplEx':
        return 'bash ./kge/kge.sh train {} {} {} {} {} {} {} {} {} {} {} {} {} -de -dr -r {}'.format(model, dataset, gpu, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk, kge_reg)

def cmd_mln(main_path, workspace_path=None, preprocessing=False):
    if preprocessing == True:
        return './mln/mln -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(main_path, main_path, main_path, mln_threshold_of_rule, mln_threads)
    else:
        return './mln/mln -load {}/mln_saved.txt -probability {}/annotation.txt -out-prediction {}/pred_mln.txt -out-rule {}/rule.txt -thresh-triplet 1 -iterations {} -lr {} -threads {}'.format(main_path, workspace_path, workspace_path, workspace_path, mln_iters, mln_lr, mln_threads)

def save_cmd(save_path):
    with open(save_path, 'w') as fo:
        fo.write('dataset: {}\n'.format(dataset))
        fo.write('iterations: {}\n'.format(iterations))
        fo.write('kge_model: {}\n'.format(kge_model))
        fo.write('kge_batch: {}\n'.format(kge_batch))
        fo.write('kge_neg: {}\n'.format(kge_neg))
        fo.write('kge_dim: {}\n'.format(kge_dim))
        fo.write('kge_gamma: {}\n'.format(kge_gamma))
        fo.write('kge_alpha: {}\n'.format(kge_alpha))
        fo.write('kge_lr: {}\n'.format(kge_lr))
        fo.write('kge_iters: {}\n'.format(kge_iters))
        fo.write('kge_tbatch: {}\n'.format(kge_tbatch))
        fo.write('kge_reg: {}\n'.format(kge_reg))
        fo.write('mln_threshold_of_rule: {}\n'.format(mln_threshold_of_rule))
        fo.write('mln_threshold_of_triplet: {}\n'.format(mln_threshold_of_triplet))
        fo.write('mln_iters: {}\n'.format(mln_iters))
        fo.write('mln_lr: {}\n'.format(mln_lr))
        fo.write('mln_threads: {}\n'.format(mln_threads))
        fo.write('weight: {}\n'.format(weight))


logger1.info('Start training')
time = str(datetime.now()).replace(' ', '_')
path = path + '/' + dataset.split('/')[-1]
ensure_dir(path)
save_cmd('{}/cmd.txt'.format(path))

# ------------------------------------------

os.system('cp {}/train.txt {}/train.txt'.format(dataset, path))
os.system('cp {}/train.txt {}/train_augmented.txt'.format(dataset, path))
logger1.info('Starting cmd_mln...')
os.system(cmd_mln(path, preprocessing=True))
logger1.info('End of cmd_mln...')
for k in range(iterations):
    logger1.info(f'### Iter {k}')
    workspace_path = path + '/' + str(k)


    ensure_dir(workspace_path)
    os.system('cp {}/train_augmented.txt {}/train_kge.txt'.format(path, workspace_path))
    os.system('cp {}/hidden.txt {}/hidden.txt'.format(path, workspace_path))
    logger1.info('Starting cmd_kge')
    os.system(cmd_kge(workspace_path, kge_model)) 
    logger1.info('cmd_kge finished!!!')


    logger1.info('Starting cmd_mln')
    os.system(cmd_mln(path, workspace_path, preprocessing=False))
    print('cmd_mln finished!!!')
    augment_triplet('{}/pred_mln.txt'.format(workspace_path), '{}/train.txt'.format(path), '{}/train_augmented.txt'.format(workspace_path), mln_threshold_of_triplet)
    os.system('cp {}/train_augmented.txt {}/train_augmented.txt'.format(workspace_path, path))

    evaluate('{}/pred_mln.txt'.format(workspace_path), '{}/pred_kge_valid.txt'.format(workspace_path), '{}/result_kge_mln.txt'.format(workspace_path), weight, count_info_dict, entity2id, logger1, args, 'val', k)
    evaluate('{}/pred_mln.txt'.format(workspace_path), '{}/pred_kge.txt'.format(workspace_path), '{}/result_kge_mln.txt'.format(workspace_path), weight, count_info_dict, entity2id, logger1, args, 'tst', k)
    

logger1.info('End of training')