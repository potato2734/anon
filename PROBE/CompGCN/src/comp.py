import os
import torch
from collections import defaultdict
from pykeen.pipeline import pipeline
from pykeen.models import CompGCN
from pykeen.datasets import FB15k237, WN18RR
from pykeen.training import LCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.typing import MappedTriples
from pykeen.utils import set_random_seed
from utils import *
from typing import Optional
import logging
import argparse
from torch.optim import Adam
import numpy as np
import random
import json
from pykeen.constants import PYKEEN_CHECKPOINTS
from probe import PROBE

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class TargetEntityRankEvaluator(RankBasedEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_ranks = defaultdict(list)  

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: str,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:
        super().process_scores_(hrt_batch, target, scores, true_scores, dense_positive_mask)
        if target == "head":  
            target_entities = hrt_batch[:, 0].cpu().tolist() 
        elif target == "tail":  
            target_entities = hrt_batch[:, 2].cpu().tolist()  
        else:
            return
        
        for i, entity in enumerate(target_entities):
            rank = (scores[i] >= true_scores[i]).sum().item()  
            self.target_ranks[entity].append(rank)  

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_layer', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--seed', type=int)

    return parser.parse_args(args)

def set_logger(args):
    log_file = os.path.join(f'./logs/', f'CompGCN_{args.data}_{args.seed}.log')

    if not os.path.exists(f'./logs/'):
        os.makedirs(f'./logs/', exist_ok=True)

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return log_file  

import torch

def get_discarded_test_triples(args, dataset):
    """
    Retrieve triples that were discarded from the test set due to missing entities/relations.
    
    Args:
        dataset: a PyKEEN Dataset object (already loaded)
        
    Returns:
        discarded_triples: list of discarded triples (head, relation, tail) as strings
    """
    original_test_triples = read_triples(f'./data/{args.data}/test.txt')
   
    mapped_triples = dataset.testing.mapped_triples.cpu().numpy()

    id_to_entity = {v: k for k, v in dataset.training.entity_to_id.items()}
    id_to_relation = {v: k for k, v in dataset.training.relation_to_id.items()}

    mapped_triples_text = []
    for h_id, r_id, t_id in mapped_triples:
        head = id_to_entity.get(int(h_id)) 
        relation = id_to_relation.get(int(r_id))
        tail = id_to_entity.get(int(t_id))
        if head is not None and relation is not None and tail is not None:
            mapped_triples_text.append((head, relation, tail)) 

    mapped_triples_set = set(mapped_triples_text)
    discarded_triples = [trip for trip in original_test_triples if trip not in mapped_triples_set]

    return discarded_triples



def main(args):
    set_random_seeds(args.seed)
    set_logger(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.data == 'FB15k237':
        dataset = FB15k237(create_inverse_triples=True)
    if args.data == 'wn18rr':
        dataset = WN18RR(create_inverse_triples=True)

    discarded_triples = get_discarded_test_triples(args, dataset)
    logging.info(discarded_triples)
    print(f"Number of discarded triples: {len(discarded_triples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model_kwargs = {
        'embedding_dim': args.dim,  # Set the embedding dimension
        'encoder_kwargs': {
            'num_layers': args.num_layer,  # Number of GCN layers
            'dims': [args.dim]*args.num_layer,  # Dimensions for each layer
            'layer_kwargs': {
                'dropout': args.dropout,  # Dropout rate for each layer
                # Add other layer-specific parameters as needed
            },
            'entity_representations_kwargs': {
                'embedding_dim': args.dim,
                # Add other entity representation-specific parameters as needed
            },
            'relation_representations_kwargs': {
                'embedding_dim': args.dim,
                # Add other relation representation-specific parameters as needed
            },
        },
        'interaction': 'distmult',  # Interaction function to use
        # Add other model-specific parameters as needed
    }
    
    model = CompGCN(triples_factory=dataset.training, **model_kwargs).to(device)
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    training_loop = LCWATrainingLoop(
        model=model,
        triples_factory=dataset.training,
        optimizer=optimizer,
    )

    early_stopper = EarlyStopper(
        model=model,
        evaluator=RankBasedEvaluator(),
        training_triples_factory=dataset.training,  
        evaluation_triples_factory=dataset.validation,  
        evaluation_batch_size=8,  
        patience=50, 
        frequency=10,  
        metric="mean_reciprocal_rank",  
        relative_delta=0.001, 
        larger_is_better=True, 
        use_tqdm=True,  
    )
    

    def print_validation_mrr(stopper, mrr, epoch):
        """Callback function to print validation MRR after each evaluation step."""
        logging.info(f"Epoch {epoch}: Validation MRR = {mrr:.4f}")

    early_stopper.continue_callbacks.append(print_validation_mrr)

    training_loop.train(
        triples_factory=dataset.training,
        num_epochs=500, 
        batch_size=args.batch,
        stopper=early_stopper, 
        use_tqdm=True, 
        
    )
    if early_stopper.results:
        best_mrr = max(early_stopper.results)
        logging.info(f"\nBest Validation MRR: {best_mrr:.10f}")
    logging.info('finishing training')
 
    evaluator = TargetEntityRankEvaluator()
    
    logging.info(f'num of mapped triples : {len(dataset.testing.mapped_triples)}')

    evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
        batch_size=8, 
        slice_size=None,  
        use_tqdm=True, 
        device=device  
    )
    target_entity_ranks = evaluator.target_ranks

    print("\nTarget Entity Predicted Ranks (First 5 Entities):")
    total_predictions = 0
    for entity, ranks in list(target_entity_ranks.items()):
        total_predictions += len(ranks)
    print(total_predictions)

    e2id = read_entities(f'./data/{args.data}/entities.dict')
    r2id = read_relations(f'./data/{args.data}/relations.dict')
    pykeen_id2id = {idx:e2id[e] for e, idx in dataset.entity_to_id.items()}
    eid2count = {}
    print(len(e2id))
    for i in range(len(e2id)):
        eid2count[i] = 0
    for _triple in read_triple(f'data/{args.data}/train.txt', e2id, r2id):
        h, r, t = _triple
        eid2count[h] += 1
        eid2count[t] += 1
 
    count_info_dict_trn = dict(sorted(eid2count.items(), key=lambda item : -item[1]))
    
    rank_dict = {}
    
    for entity, ranks in list(target_entity_ranks.items()):
        rank_dict[pykeen_id2id[entity]] = ranks
        
    # triples that is composed of unseen entities during training recieves expected rank(=(1+nentity)/2)
    for h, r, t in discarded_triples:
        try:
            rank_dict[e2id[h]].append((1 + len(e2id)) / 2)
        except:
            rank_dict[e2id[h]] = []
            rank_dict[e2id[h]].append((1 + len(e2id)) / 2)
            
        try:
            rank_dict[e2id[t]].append((1 + len(e2id)) / 2)
        except:
            rank_dict[e2id[t]] = []
            rank_dict[e2id[t]].append((1 + len(e2id)) / 2)
        
    count_info_dict_tst = {k: len(v) for k, v in rank_dict.items()}
    met = PROBE(rank_dict, count_info_dict_trn, count_info_dict_tst, len(eid2count))
    met.set_raw_ranks(rank_dict)
    alphas, betas = [0.25, 0.5, 1.0, 2.0], [0.8, 0.4, 0.2, 0.0]
    for a in alphas:
        for b in betas:
            logging.info(f'PROBE(alpha={a},beta={b}) : {met.calculate_final_metric(a, b)}')

if __name__ == '__main__':
    main(parse_args())


