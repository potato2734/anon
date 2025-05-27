import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
from easydict import EasyDict
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset
from predictors import Predictor, PredictorPlus
from generators import Generator
from utils import load_config, save_config, set_logger, set_seed, save_model, load_model
from trainer import TrainerPredictor, TrainerGenerator
import comm

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='RNNLogic',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument("--config", default='../rnnlogic.yaml', type=str)
    parser.add_argument("--smoothing_p", type=float)
    parser.add_argument("--rule_file", type=str)
    parser.add_argument("--smoothing_pp", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(args)

def main(args): 
    cfgs = load_config(args.config)
    cfg = cfgs[0] # cfg.chunk contains the chunk cnt information

    if cfg.save_path is None:
        cfg.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    
    if cfg.save_path and not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)

    cfg.predictor.train.smoothing = args.smoothing_p
    cfg.predictorplus.train.smoothing = args.smoothing_pp
    
    save_config(cfg, cfg.save_path)

    set_logger(args, cfg.save_path)
    
    
    cfg.data.rule_file = args.rule_file
    set_seed(args.seed)

    graph = KnowledgeGraph(cfg.data.data_path)

    train_set = TrainDataset(graph, cfg.data.batch_size)
    valid_set = ValidDataset(graph, cfg.data.batch_size)
    test_set = TestDataset(graph, cfg.data.batch_size)

    dataset = RuleDataset(graph.relation_size, cfg.data.rule_file)

    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Pre-train Generator')
        logging.info('-------------------------')
    generator = Generator(graph, **cfg.generator.model)
    solver_g = TrainerGenerator(generator, gpu=cfg.generator.gpu)
    solver_g.train(dataset, **cfg.generator.pre_train)

    replay_buffer = list()
    eid2count = {}
    for i in range(graph.entity_size):
        eid2count[i] = 0

    for _triple in graph.train_facts:
        h, r, t = _triple
        eid2count[h] += 1
        eid2count[t] += 1
    count_info_dict_trn = dict(sorted(eid2count.items(), key=lambda item : -item[1]))
    for k in range(cfg.EM.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| EM Iteration: {}/{}'.format(k + 1, cfg.EM.num_iters))
            logging.info('-------------------------')
        
        # Sample logic rules.
        sampled_rules = solver_g.sample(cfg.EM.num_rules, cfg.EM.max_length)
        prior = [rule[-1] for rule in sampled_rules]
        rules = [rule[0:-1] for rule in sampled_rules]

        # Train a reasoning predictor with sampled logic rules.
        predictor = Predictor(graph, **cfg.predictor.model)
        predictor.set_rules(rules)
        optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)

        solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, gpus=cfg.predictor.gpus)
        solver_p.train(**cfg.predictor.train)
        valid_mrr_iter = solver_p.evaluate(count_info_dict_trn, eid2count, 'valid', expectation=cfg.predictor.eval.expectation, args=args)
        # save_model(solver_p, optim, cfg)
        test_mrr_iter = solver_p.evaluate(count_info_dict_trn, eid2count, 'test', expectation=cfg.predictor.eval.expectation, args=args)
        
        # E-step: Compute H scores of logic rules.
        likelihood = solver_p.compute_H(**cfg.predictor.H_score)
        posterior = [l + p * cfg.EM.prior_weight for l, p in zip(likelihood, prior)]
        for i in range(len(rules)):
            rules[i].append(posterior[i])
        replay_buffer += rules
        
        # M-step: Update the rule generator.
        dataset = RuleDataset(graph.relation_size, rules)
        solver_g.train(dataset, **cfg.generator.train)
        
    
        
    if replay_buffer != []:
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| Post-train Generator')
            logging.info('-------------------------')
        dataset = RuleDataset(graph.relation_size, replay_buffer)
        solver_g.train(dataset, **cfg.generator.post_train)

    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Beam Search Best Rules')
        logging.info('-------------------------')
    
    sampled_rules = list()
    for num_rules, max_length in zip(cfg.final_prediction.num_rules, cfg.final_prediction.max_length):
        sampled_rules_ = solver_g.beam_search(num_rules, max_length)
        sampled_rules += sampled_rules_
        
    prior = [rule[-1] for rule in sampled_rules]
    rules = [rule[0:-1] for rule in sampled_rules]

    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Train Final Predictor+')
        logging.info('-------------------------')

    predictor = PredictorPlus(graph, **cfg.predictorplus.model)
    predictor.set_rules(rules)
    optim = torch.optim.Adam(predictor.parameters(), **cfg.predictorplus.optimizer)

    solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, gpus=cfg.predictorplus.gpus)
    best_valid_mrr = 0.0
    test_mrr_iter = 0.0
    for k in range(cfg.final_prediction.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| Iteration: {}/{}'.format(k + 1, cfg.final_prediction.num_iters))
            logging.info('-------------------------')
            

        solver_p.train(**cfg.predictorplus.train)
        valid_mrr_iter = solver_p.evaluate(count_info_dict_trn, eid2count, 'valid', expectation=cfg.predictorplus.eval.expectation, args=args)
        

        if valid_mrr_iter > best_valid_mrr:
            best_valid_mrr = valid_mrr_iter
            test_mrr_iter = solver_p.evaluate(count_info_dict_trn, eid2count, 'test', expectation=cfg.predictorplus.eval.expectation, args=args)

    
    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Final Best Valid MRR: {:.6f}'.format(best_valid_mrr))
        logging.info('-------------------------')

        logging.info('-------------------------')
        logging.info('| Final Best Test MRR: {:.6f}'.format(test_mrr_iter))
        logging.info('-------------------------')

if __name__ == '__main__':
    main(parse_args())